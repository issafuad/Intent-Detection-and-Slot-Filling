__author__ = 'fuadissa'

import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
DATASET_FOLDER = os.path.join(DATA_FOLDER, 'dataset')
MODEL_SERVING_DIR = os.path.join(PROJECT_ROOT, 'model_served')
TRAINING_DATASET = os.path.join(DATASET_FOLDER, 'atis.train.pkl')
TESTING_DATASET = os.path.join(DATASET_FOLDER, 'atis.test.pkl')
WORD2VEC = os.path.join(DATA_FOLDER, 'GoogleNews-vectors-negative300.bin')
TRAINED_MODELS_PATH = os.path.join(DATA_FOLDER, 'models')
CHECKPOINT = 'ckpt'
TENSORBOARD_FOLDER = 'tb'

TRAINING_SETTINGS = 'training_settings.pkl'
VOCAB2ID = 'vocab2id.pkl'
SLOT2ID = 'slot2id.pkl'
INTENT2ID = 'intent2id.pkl'