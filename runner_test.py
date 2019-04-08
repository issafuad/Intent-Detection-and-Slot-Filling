__author__ = 'fuadissa'

import argparse

from atis_classifer.classifier import Classifier
from settings import TESTING_DATASET

def test_classifier(testing_settings):
    model_path = testing_settings['model_path']
    model = Classifier(model_path)
    model.test_model(TESTING_DATASET)


def get_arguments():
    parser = argparse.ArgumentParser(description='Testing parameters')
    parser.add_argument('model_path', type=str)
    return vars(parser.parse_args())

if __name__ == '__main__':
    testing_settings = get_arguments()
    test_classifier(testing_settings)