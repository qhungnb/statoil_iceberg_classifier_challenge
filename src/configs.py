import os

BASE_DIR = 'E:\python3\statoil_iceberg_classifier_challenge'

MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
MODEL_FILE = os.path.join(MODEL_DIR, 'model_fold_{}.h5')

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
TEST_FILE = os.path.join(DATA_DIR, 'test.json')

RESULT_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'outputs')
RESULT_FILE = os.path.join(OUTPUT_DIR, 'submission_{}.csv')
