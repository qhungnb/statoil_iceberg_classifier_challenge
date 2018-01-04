import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from random import random
from time import time
from data_loader import DataLoader
import models
import configs
import utils

from keras.preprocessing.image import (
    random_rotation,
    random_shift,
    random_zoom,
    ImageDataGenerator
)
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau
)

P = 0.5
EPOCHS = 40
BATCH_SIZE = 32
AUGMENTS = {
    'rotation': {'rg': 45, 'row_axis': 0, 'col_axis': 1, 'channel_axis': 2, 'fill_mode': 'nearest'},
    'shift': {'wrg': 0.1, 'hrg': 0.5, 'row_axis': 0, 'col_axis': 1, 'channel_axis': 2},
    'shear': {'intensity': 0.1, 'row_axis': 0, 'col_axis': 1, 'channel_axis': 2},
    'zoom': {'zoom_range': (0.1, 0.1), 'row_axis': 0, 'col_axis': 1, 'channel_axis': 2},
    'ho_flip': True,
    'ver_flip': True
}


def get_callbacks(model_name='ice'):
    # Create callback to save best model while training
    checkpoint = ModelCheckpoint(
        filepath=configs.MODEL_FILE.format(model_name),
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        patience=15,
        verbose=1
    )

    tensor_board = TensorBoard(
        log_dir='logs/{}'.format(model_name),
        write_grads=True
    )

    reduce_lr = ReduceLROnPlateau(
        factor=0.25,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        epsilon=1e-4
    )
    return [checkpoint, early_stopping, tensor_board, reduce_lr]


def get_processed_image(image):
    if random() > P and AUGMENTS['ho_flip']:
        image = cv2.flip(image, 1)

    if random() > P and AUGMENTS['ver_flip']:
        image = cv2.flip(image, 0)

    if random() > P:
        image = random_rotation(image, **AUGMENTS['rotation'])

    if random() > P:
        image = random_shift(image, **AUGMENTS['shift'])

    if random() > P:
        image = random_zoom(image, **AUGMENTS['zoom'])

    return image


def generator(images, angles, labels, shuffle=True, batch_size=32):
    while True:
        # Generate order of samples
        indices = np.arange(len(labels))
        if shuffle:
            np.random.shuffle(indices)

        # Generate batches
        for start_id in range(0, len(labels), batch_size):
            batch_ids = indices[start_id: start_id + batch_size]

            batch_images = np.array([
                get_processed_image(image)
                for image in images[batch_ids]
            ])

            batch_angles = angles[batch_ids]
            batch_labels = labels[batch_ids]
            yield [batch_images, batch_angles], batch_labels


# Define the image transformations here
data_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0,
    zoom_range=0.5,
    rotation_range=10
)


# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(images, angles, labels, batch_size=32, seed=55):
    gen_images = data_generator.flow(images, labels, batch_size=batch_size, seed=seed)
    gen_angles = data_generator.flow(images, angles, batch_size=batch_size, seed=seed)
    while True:
            batch_images_labels = gen_images.next()
            batch_images_angles = gen_angles.next()
            yield [batch_images_labels[0], batch_images_angles[1]], batch_images_labels[1]


def train():
    data_loader = DataLoader(
        data_file=configs.TRAIN_FILE
    )
    images, angles, labels = data_loader.get_data

    print('Image shape: {}'.format(images.shape))
    print('Angles shape: {}'.format(angles.shape))
    print('Labels shape: {}'.format(labels.shape))

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.16, random_state=1024)

    for idx, [train_ids, val_ids] in enumerate(sss.split(images, labels)):
        train_images, val_images = images[train_ids], images[val_ids]
        train_angles, val_angles = angles[train_ids], angles[val_ids]
        train_labels, val_labels = labels[train_ids], labels[val_ids]

        if os.path.isfile(configs.MODEL_FILE.format(idx)):
            model = load_model(configs.MODEL_FILE.format(idx))
        else:
            model = models.Resnet(
                input_shape=data_loader.image_shape
            )
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit_generator(
            generator=gen_flow_for_two_inputs(train_images, train_angles, train_labels, batch_size=BATCH_SIZE),
            steps_per_epoch=np.ceil(8 * len(train_ids) / BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=([val_images, val_angles], val_labels),
            callbacks=get_callbacks(model_name=str(idx)),
            use_multiprocessing=False,
            workers=1
        )

        print('Ford: {}'.format(idx))
        model = load_model(configs.MODEL_FILE.format(idx))

        p = model.predict([train_images, train_angles], batch_size=BATCH_SIZE, verbose=1)
        print('\nEvaluate loss on training data: {}'.format(metrics.log_loss(train_labels, p)), flush=True)

        p = model.predict([val_images, val_angles], batch_size=BATCH_SIZE, verbose=1)
        print('\nEvaluate loss on validation data: {}'.format(metrics.log_loss(val_labels, p)), flush=True)


def get_result(model_file=None, test_data=None, model_name=None):
    model = load_model(model_file)

    prob = model.predict(
        x=[test_data.images, test_data.inc_angle],
        batch_size=BATCH_SIZE,
        verbose=1
    )
    results = np.squeeze(prob, axis=-1)

    submission = pd.DataFrame({'id': test_data.ids, 'is_iceberg': results})
    submission.to_csv(configs.RESULT_FILE.format(
        '{}_{}'.format(utils.get_file_name(model_file), model_name),
        index=False
    ))

    return results


def get_final_results(model_dir=None, model_name='final'):
    print('\tLoading test data...')
    test_data = DataLoader(data_file=configs.TEST_FILE)
    print('\tDone.')

    print('\tGetting submissions...')
    results = []
    for model_file in glob(os.path.join(model_dir, '*.h5')):
        result = get_result(model_file, test_data, model_name)
        results.append(result)

    results = np.mean(results, axis=0)

    submission = pd.DataFrame({'id': test_data.ids, 'is_iceberg': results})
    submission.to_csv(configs.RESULT_FILE.format(model_name), index=False)

    print('\tDone.')


if __name__ == '__main__':
    train()
    # get_final_results(model_dir=configs.MODEL_DIR, model_name='resnet')
