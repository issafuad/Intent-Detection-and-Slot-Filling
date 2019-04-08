__author__ = 'fuadissa'

import os
import logging

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

from atis_classifer.model_graph import build_graph
from settings import CHECKPOINT, TENSORBOARD_FOLDER

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)


class NetworkTrainer(object):
    def __init__(self, training_settings):
        self.training_settings = training_settings
        self.graph = build_graph(training_settings)
        self.summary_writer = None
        self.saver = None
        self.session = None

    def add_metric_summaries(self, mode, iteration, name2metric):
        """Add summary for metric."""
        metric_summary = tf.Summary()
        for name, metric in name2metric.items():
            metric_summary.value.add(tag='{}_{}'.format(mode, name), simple_value=metric)
        self.summary_writer.add_summary(metric_summary, global_step=iteration)

    def show_train_stats(self, epoch, iteration, losses, y_true_intent, y_pred_intent, y_true_slot, y_pred_slot):
        # compute mean statistics
        loss = np.mean(losses)
        accuracy_intent = accuracy_score(y_true_intent, y_pred_intent)
        accuracy_slot = np.mean([accuracy_score(seq_slot_true, seq_slot_pred) for seq_slot_true, seq_slot_pred in
                                 zip(y_true_slot, y_pred_slot)])
        LOGGER.info(
            'Epoch={}, Iter={:,}, Mean Training Loss={:.4f}, Accuracy={:.4f}, Slot Accuracy={:.4f}'.format(epoch,
                                                                                                           iteration,
                                                                                                           loss,
                                                                                                           accuracy_intent,
                                                                                                           accuracy_slot))
        self.add_metric_summaries('train', iteration,
                                  {'cross_entropy': loss, 'accuracy': accuracy_intent, 'slot_accuracy': accuracy_slot})
        LOGGER.info('\n{}'.format(classification_report(y_true_intent, y_pred_intent, digits=3)))

    #        LOGGER.info('\n{}'.format(classification_report(y_true_slot, y_pred_slot, digits=3)))

    def validate(self, epoch, iteration, batcher, best_loss, patience):
        """Validate the atis_classifer on validation set."""

        losses, y_intent_true, y_intent_pred, y_slot_true, y_slot_pred = list(), list(), list(), list(), list()

        for (X_batch_sent, y_true_slot_batch, y_true_intent_batch, seq_length), _ in batcher:
            y_pred_intent_batch, y_pred_slot_batch, loss_batch = self.session.run(
                [self.graph.get_tensor_by_name('intent/y_pred:0'),
                 self.graph.get_tensor_by_name('slot/y_pred:0'),
                 self.graph.get_tensor_by_name('loss/loss:0')],
                feed_dict={
                    'inputs/x_sent:0': X_batch_sent,
                    'inputs/sequence_length:0': seq_length,
                    'inputs/y_slot:0': y_true_slot_batch,
                    'inputs/y_intent:0': y_true_intent_batch,
                    'inputs/dropout:0': 1
                })
            losses.append(loss_batch)
            y_slot_true.extend(y_true_slot_batch)
            y_slot_pred.extend(y_pred_slot_batch)
            y_intent_pred.extend(y_pred_intent_batch)
            y_intent_true.extend(y_true_intent_batch)

        # compute mean statistics
        loss = np.mean(losses)
        accuracy = accuracy_score(y_intent_true, y_intent_pred)
        accuracy_slot = np.mean([accuracy_score(seq_slot_true, seq_slot_pred) for seq_slot_true, seq_slot_pred in
                                 zip(y_slot_true, y_slot_pred)])

        LOGGER.info(
            'Epoch={}, Iter={:,}, Validation Loss={:.4f}, Intent Accuracy={:.4f}, Slot Accuracy={:.4f}'.format(epoch,
                                                                                                               iteration,
                                                                                                               loss,
                                                                                                               accuracy,
                                                                                                               accuracy_slot))
        self.add_metric_summaries('valid', iteration, {'cross_entropy': loss, 'validation_accuracy': accuracy,
                                                       'validation_intent_accuracy': accuracy_slot})
        LOGGER.info('\n{}'.format(classification_report(y_intent_true, y_intent_pred, digits=3)))

        if loss < best_loss:
            LOGGER.info('Best score Loss so far, save the atis_classifer.')
            self.save()
            best_loss = loss

            if iteration * 2 > patience:
                patience = iteration * 2
                LOGGER.info('Increased patience to {:,}'.format(patience))
        return best_loss, patience

    def save(self):
        self.saver.save(self.session, os.path.join(self.training_settings['model_path'], CHECKPOINT))
        LOGGER.info('Finished Saving')

    def train_network(self, train_batcher, valid_batcher, embedding, train_number_of_instances):

        self.graph = build_graph(self.training_settings)
        pretrained_embeddings = embedding[self.training_settings['reserved_vocab_length']:]
        patience = self.training_settings['patience']
        best_valid_loss = np.float64('inf')
        with tf.Session(graph=self.graph) as self.session:

            self.summary_writer = tf.summary.FileWriter(
                os.path.join(self.training_settings['model_path'], TENSORBOARD_FOLDER),
                self.session.graph)
            self.saver = tf.train.Saver(name='saver')

            self.session.run(tf.global_variables_initializer())
            if self.training_settings['use_pretrained_embeddings']:
                self.session.run(self.graph.get_operation_by_name('embedding/assign_pretrained_embeddings'),
                                 feed_dict={'inputs/pretrained_embeddings_ph:0': pretrained_embeddings})

            batches_in_train = train_number_of_instances / self.training_settings['batch_size']
            train_stat_interval = max(batches_in_train // self.training_settings['train_interval'], 1)
            valid_stat_interval = max(batches_in_train // self.training_settings['valid_interval'], 1)

            losses, y_intent_true, y_intent_pred, y_slot_true, y_slot_pred = list(), list(), list(), list(), list()
            for batch_num, ((X_batch_sent, y_true_slot_batch, y_true_intent_batch, seq_length), new_start) in enumerate(
                    train_batcher):
                iteration = batch_num * self.training_settings['batch_size']
                epoch = 1 + iteration // train_number_of_instances
                if new_start:
                    losses, y_intent_true, y_intent_pred, y_slot_true, y_slot_pred = list(), list(), list(), list(), list()

                if iteration > train_number_of_instances * self.training_settings['max_epoch']:
                    LOGGER.info('reached max epoch')
                    break

                _, y_pred_intent_batch, y_pred_slot_batch, loss_batch = self.session.run(
                    [self.graph.get_operation_by_name('optimizer/optimizer'),
                     self.graph.get_tensor_by_name('intent/y_pred:0'),
                     self.graph.get_tensor_by_name('slot/y_pred:0'),
                     self.graph.get_tensor_by_name('loss/loss:0')],
                    feed_dict={
                        'inputs/x_sent:0': X_batch_sent,
                        'inputs/sequence_length:0': seq_length,
                        'inputs/y_slot:0': y_true_slot_batch,
                        'inputs/y_intent:0': y_true_intent_batch,
                        'inputs/dropout:0': self.training_settings['dropout']
                    }
                )

                losses.append(loss_batch)
                y_slot_true.extend(y_true_slot_batch)
                y_slot_pred.extend(y_pred_slot_batch)
                y_intent_pred.extend(y_pred_intent_batch)
                y_intent_true.extend(y_true_intent_batch)

                if batch_num % train_stat_interval == 0:
                    self.show_train_stats(epoch, iteration, losses, y_intent_true, y_intent_pred, y_slot_true,
                                          y_slot_pred)

                if batch_num % valid_stat_interval == 0:
                    best_valid_loss, patience = self.validate(epoch, iteration, valid_batcher, best_valid_loss,
                                                              patience)

                if iteration > patience:
                    LOGGER.info('Iteration is more than patience, finish training.')
                    break

            LOGGER.info('Finished fitting the atis_classifer.')
            LOGGER.info('Best Validation Cross-entropy Loss: {:.4f}'.format(best_valid_loss))
