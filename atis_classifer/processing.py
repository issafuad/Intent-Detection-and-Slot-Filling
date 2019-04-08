__author__ = 'fuadissa'


import logging
from collections import Counter

import numpy as np
import pickle
from gensim.models import KeyedVectors

from settings import WORD2VEC

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)

START_OF_SENTENCE = '__bos__'
END_OF_SENTENCE = '__eos__'
UNKNOWN_WORD = '__unknown__'
PADDING_WORD = '__padded__'


class VocabProcessor(object):
    PROCESSOR_RESERVED = [START_OF_SENTENCE, END_OF_SENTENCE, UNKNOWN_WORD, PADDING_WORD]

    def __init__(self, vocab2id=None):
        self.vocab2id = vocab2id if vocab2id else dict()

    def get_word2vec(self, vocab: list, reserved_vocab=list()) -> tuple:
        vocab_set = set(vocab)
        reserved_vocab = self.PROCESSOR_RESERVED + reserved_vocab
        reserved_vocab_set = set(reserved_vocab)

        LOGGER.info('Loading Word2Vec Vocabulary')
        # TODO add option not to get word embeddings
        word2vec = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
        LOGGER.info('Finished loading Word2Vec Vocabulary. {} loaded'.format(word2vec.vectors.shape[0]))
        chosen_word_index_set = set()
        for index, each_word in enumerate(word2vec.index2word):

            if not index % 100000:
                LOGGER.info('checked {}/{}'.format(index, len(word2vec.index2word)))

            if each_word in vocab_set and each_word not in reserved_vocab_set:
                chosen_word_index_set.add(index)

        LOGGER.info('Found {}/{} of dataset vocab'.format(len(chosen_word_index_set), len(vocab_set)))
        chosen_word_index_list = list(chosen_word_index_set)
        embedding_vocab = [word2vec.index2word[each] for each in chosen_word_index_list]
        all_vocab = reserved_vocab + embedding_vocab
        self.vocab2id = {vocab: index for index, vocab in enumerate(all_vocab)}
        word_embeddings = word2vec.vectors[chosen_word_index_list]
        reserved_embeddings = np.random.random((len(reserved_vocab), word_embeddings.shape[1]))
        word_embeddings_all = np.concatenate((reserved_embeddings, word_embeddings), axis=0)

        return word_embeddings_all, embedding_vocab, reserved_vocab

    def transform(self, X: list) -> list:
        transformed_X = list()

        for query in X:
            transformed_words = list()

            for each_word in query:

                if each_word == 'EOS':
                    transformed_words.append(self.vocab2id.get(END_OF_SENTENCE))
                    continue

                if each_word == 'BOS':
                    transformed_words.append(self.vocab2id.get(START_OF_SENTENCE))
                    continue

                transformed_words.append(
                    self.vocab2id.get(each_word, self.vocab2id.get(each_word.lower(), self.vocab2id.get(UNKNOWN_WORD))))

            transformed_X.append(transformed_words)

        return transformed_X

    def pad(self, X: list, padding_size: int) -> tuple:

        padded_X = list()
        seq_length = list()
        sentences_cut_counter = 0
        for each_sent in X:
            if len(each_sent) <= padding_size:
                padded_sent = each_sent + [self.vocab2id[PADDING_WORD]] * (padding_size - len(each_sent))
            else:
                padded_sent = each_sent[:padding_size]
                sentences_cut_counter += 1

            padded_X.append(padded_sent)
            seq_length.append(each_sent.index(self.vocab2id[END_OF_SENTENCE]))

        LOGGER.info('number of shortened sentences : {}'.format(sentences_cut_counter))

        return padded_X, seq_length


def convert_with_mapping(list_of_lists: list, mapping: dict) -> list:
    new_list_of_lists = list()
    for each_list in list_of_lists:
        token_list = list()

        for each in each_list:
            token_list.append(mapping[each])

        new_list_of_lists.append(token_list)
    return new_list_of_lists


def add_padding_id(mapping: dict) -> tuple:
    padding_id = max(mapping.values()) + 1
    mapping[PADDING_WORD] = padding_id
    return mapping, padding_id


def pad(X: list, padding_size: int, padding_id: int) -> list:
    padded_X = list()
    sentences_cut_counter = 0
    for each_sent in X:
        each_sent = each_sent.tolist()
        if len(each_sent) <= padding_size:
            padded_sent = each_sent + [padding_id] * (padding_size - len(each_sent))
        else:
            padded_sent = each_sent[:padding_size]
            sentences_cut_counter += 1

        padded_X.append(padded_sent)

    return padded_X


def get_dataset(file_path: str) -> tuple:
    with open(file_path, 'rb') as stream:
        ds, dicts = pickle.load(stream)

    print('Done  loading: ', file_path)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))

    y_intent = [each[0] for each in ds['intent_labels']]

    X = ds['query']
    y_slot = ds['slot_labels']
    token2id = dicts['token_ids']
    intent2id = dicts['intent_ids']
    slot2id = dicts['slot_ids']
    id2token = {token2id[k]: k for k in token2id}
    X = convert_with_mapping(X, id2token)

    counts = Counter(y_intent)

    for label, count in counts.items():
        if count == 1:
            index_to_remove = y_intent.index(label)
            del y_intent[index_to_remove]
            del y_slot[index_to_remove]
            del X[index_to_remove]

    return X, y_slot, y_intent, token2id, slot2id, intent2id


def batcher(lists_to_batch: list, batch_size: int, infinite=False) -> tuple:
    length_of_list = len(lists_to_batch[0])
    start_index = 0
    while True:
        new_start = False
        batched_lists = list()
        if start_index + batch_size < length_of_list:
            end_index = start_index + batch_size
        else:
            end_index = length_of_list
            new_start = True

        for list_to_batch in lists_to_batch:
            batched_lists.append(list_to_batch[start_index: end_index])

        if new_start:
            start_index = 0
            if not infinite:
                break
        else:
            start_index += batch_size

        yield tuple(batched_lists), new_start
    yield tuple(batched_lists), new_start
