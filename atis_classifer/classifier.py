__author__ = 'fuadissa'

import pickle
import os
import logging

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize

from atis_classifer.processing import VocabProcessor
from atis_classifer.processing import batcher, get_dataset, pad
from atis_classifer.processing import PADDING_WORD, START_OF_SENTENCE, END_OF_SENTENCE
from settings import VOCAB2ID, INTENT2ID, SLOT2ID, TRAINING_SETTINGS, DATA_FOLDER

TEST_BATCH_SIZE = 128

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)

class Classifier(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.training_settings = pickle.load(open(os.path.join(model_path, TRAINING_SETTINGS), 'rb'))
        self.vocab2id = pickle.load(open(os.path.join(model_path, VOCAB2ID), 'rb'))
        self.id2vocab = {self.vocab2id[k]:k for k in self.vocab2id}
        self.intent2id = pickle.load(open(os.path.join(model_path, INTENT2ID), 'rb'))
        self.id2intent = {self.intent2id[k]:k for k in self.intent2id}
        self.slot2id = pickle.load(open(os.path.join(model_path, SLOT2ID), 'rb'))
        self.id2slot = {self.slot2id[k]:k for k in self.slot2id}
        self.vocab_processor = VocabProcessor(self.vocab2id)
        self.session = self.load_model()

    def load_model(self):
        meta_path = os.path.join(self.model_path, 'ckpt.meta')
        checkpoint_path = os.path.join(self.model_path, 'ckpt')

        graph = tf.Graph()
        session = tf.Session(graph=graph)

        with graph.as_default():
            new_saver = tf.train.import_meta_graph(meta_path)
            new_saver.restore(session, checkpoint_path)

        return session

    def predict(self, X, sequence_length):
        graph = self.session.graph

        test_batcher = batcher([X, sequence_length], TEST_BATCH_SIZE)

        y_intent_pred, y_slot_pred = list(), list()

        for (X_batch_sent, sequence_length), _ in test_batcher:
            y_pred_intent_batch, y_pred_slot_batch = self.session.run(
                [graph.get_tensor_by_name('intent/y_pred:0'),
                 graph.get_tensor_by_name('slot/y_pred:0')],
                feed_dict={
                    'inputs/x_sent:0': X_batch_sent,
                    'inputs/sequence_length:0':sequence_length,
                    'inputs/dropout:0': 1
                })

            y_slot_pred.extend(y_pred_slot_batch)
            y_intent_pred.extend(y_pred_intent_batch)

        return y_intent_pred, y_slot_pred

    def preprocess(self, X):
        X_transformed = self.vocab_processor.transform(X)
        X_padded, X_seq_length = self.vocab_processor.pad(X_transformed, self.training_settings['maximum_sent_length'])
        return X_padded, X_seq_length


    def process_text(self, text):
        tokenized = word_tokenize(text)
        processed_tokenized = [START_OF_SENTENCE] + tokenized + [END_OF_SENTENCE]
        return processed_tokenized

    def run_classifier(self, text):
        X = self.process_text(text)
        X_processed, sequence_length = self.preprocess([X])
        y_intent_pred, y_slot_pred = self.predict(X_processed, sequence_length)

        y_slot_pred = [[self.id2slot[each] for each in each_row] for each_row in y_slot_pred]
        y_intent_pred = [self.id2intent[each_row] for each_row in y_intent_pred]

        return {'slot': y_slot_pred, 'intent': y_intent_pred}

    def test_model(self, test_set_path):
        X, y_slot_true, y_intent_true, _, _, _ = get_dataset(test_set_path)
        y_slot_true = pad(y_slot_true, self.training_settings['maximum_sent_length'], self.slot2id[PADDING_WORD])

        X_processed, sequence_length = self.preprocess(X)
        y_intent_pred, y_slot_pred = self.predict(X_processed, sequence_length)

        X_sentences = [' '.join(each_row) for each_row in X]
        y_slot_true = [[self.id2slot[each] for each in each_row] for each_row in y_slot_true]
        y_slot_pred = [[self.id2slot[each] for each in each_row] for each_row in y_slot_pred]
        y_intent_true = [self.id2intent[each_row] for each_row in y_intent_true]
        y_intent_pred = [self.id2intent[each_row] for each_row in y_intent_pred]

        accuracy_intent = accuracy_score(y_intent_true, y_intent_pred)
        accuracy_slot = np.mean([accuracy_score(seq_slot_true, seq_slot_pred) for seq_slot_true, seq_slot_pred in zip(y_slot_true, y_slot_pred)])

        results = pd.DataFrame({'sentences': X_sentences,
                      'gold intent': y_intent_true,
                      'predicted intent': y_intent_pred,
                      'gold slot': y_slot_true,
                      'predicted slot': y_slot_pred})

        LOGGER.info('Intent Accuracy : {}\nSlot Accuracy : {}'.format(accuracy_intent, accuracy_slot))
        LOGGER.info('{}\n'.format(classification_report(y_intent_true, y_intent_pred, digits=3)))

        results.to_csv(os.path.join(DATA_FOLDER, 'results.csv'), encoding='utf-8')