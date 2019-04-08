__author__ = 'fuadissa'

import argparse
import pickle
import os
import logging

from sklearn.model_selection import train_test_split

from atis_classifer.processing import VocabProcessor
from atis_classifer.processing import get_dataset, batcher, add_padding_id, pad
from atis_classifer.model_trainer import NetworkTrainer
from settings import TRAINING_DATASET, TRAINING_SETTINGS, VOCAB2ID, INTENT2ID, SLOT2ID

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(training_settings: dict):
    LOGGER.info('Start Training')

    X, y_slot, y_intent, token2id, slot2id, intent2id = get_dataset(TRAINING_DATASET)
    slot2id, padding_id = add_padding_id(slot2id)
    LOGGER.info('Loaded dataset')

    vocab_processor = VocabProcessor()
    embedding, embedding_vocab, reserved_vocab = vocab_processor.get_word2vec(vocab=token2id.keys())
    training_settings['reserved_vocab_length'] = len(embedding_vocab)
    training_settings['pretrained_vocab_length'] = len(reserved_vocab)
    training_settings['intent_num'] = len(intent2id.keys())
    training_settings['slot_num'] = len(slot2id.keys())
    training_settings['maximum_sent_length'] = max([len(each) for each in X])

    X_transformed = vocab_processor.transform(X)
    X_padded, X_seq_length = vocab_processor.pad(X_transformed, training_settings['maximum_sent_length'])
    y_slot = pad(y_slot, training_settings['maximum_sent_length'], padding_id)

    X_train, X_test, y_slot_train, y_slot_test, y_intent_train, y_intent_test, seq_length_train, seq_length_test = \
        train_test_split(X_padded, y_slot, y_intent, X_seq_length, stratify=y_intent)

    train_batcher = batcher([X_train, y_slot_train, y_intent_train, seq_length_train], training_settings['batch_size'],
                            infinite=True)
    valid_batcher = batcher([X_test, y_slot_test, y_intent_test, seq_length_test], training_settings['batch_size'])
    train_number_of_instance = len(X_train)

    os.mkdir(training_settings['model_path']) if not os.path.isdir(training_settings['model_path']) else None
    pickle.dump(training_settings, open(os.path.join(training_settings['model_path'], TRAINING_SETTINGS), 'wb'))
    pickle.dump(vocab_processor.vocab2id, open(os.path.join(training_settings['model_path'], VOCAB2ID), 'wb'))
    pickle.dump(slot2id, open(os.path.join(training_settings['model_path'], SLOT2ID), 'wb'))
    pickle.dump(intent2id, open(os.path.join(training_settings['model_path'], INTENT2ID), 'wb'))

    LOGGER.info('Number of training instances : {}'.format(train_number_of_instance))
    network_trainer = NetworkTrainer(training_settings)
    network_trainer.train_network(
        train_batcher,
        list(valid_batcher),
        embedding,
        train_number_of_instance)


def get_arguments():
    parser = argparse.ArgumentParser(description='Parameters of the atis_classifer')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--use_pretrained_embeddings', nargs='?', type=str2bool, default=True)
    parser.add_argument('--embedding_size', nargs='?', type=int, default=300)
    parser.add_argument('--batch_size', nargs='?', type=int, default=128)
    parser.add_argument('--hidden_units', nargs='?', type=int, default=32)
    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.1)
    parser.add_argument('--patience', nargs='?', type=int, default=10240000)
    parser.add_argument('--train_interval', nargs='?', type=int, default=5)
    parser.add_argument('--valid_interval', nargs='?', type=int, default=2)
    parser.add_argument('--dropout', nargs='?', type=float, default=0.7)
    parser.add_argument('--max_epoch', nargs='?', type=int, default=100000)
    return vars(parser.parse_args())


if __name__ == '__main__':
    training_settings = get_arguments()
    train(training_settings)
