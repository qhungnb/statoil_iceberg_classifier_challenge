import numpy as np
import time

import utils


class DataLoader(object):
    def __init__(self, data_file=None):
        self._data_file = data_file
        self._is_train_file = (data_file.find('train') > -1)
        self._load_data()

    def _load_data(self):
        data_frame = utils.load_json(self._data_file)
        self._process(data_frame)

        if self._is_train_file:
            self.labels = data_frame['is_iceberg'].as_matrix().astype(np.float32)
        else:
            self.ids = data_frame['id'].as_matrix()

        del data_frame
        time.sleep(0.01)

    @staticmethod
    def _rescale(imgs):
        return imgs / 100. + 0.5

    def _preprocess(self):
        self.images = self._rescale(self.images)

    def _process(self, raw_data):
        self.inc_angle = raw_data['inc_angle'].replace('na', 0).as_matrix().astype(np.float32)

        band1 = np.array([np.array(band).reshape(75, 75).astype(np.float32) for band in raw_data['band_1']])
        band2 = np.array([np.array(band).reshape(75, 75).astype(np.float32) for band in raw_data['band_2']])
        band3 = band1 + band2

        self.images = np.stack([band1, band2, band3], axis=-1)

        self._preprocess()

    @property
    def image_shape(self):
        return self.images.shape[1:]

    @property
    def size(self):
        return self.images.shape[0]

    @property
    def get_data(self):
        return self.images, self.inc_angle, self.labels

    def generate_batches(self, shuffle=True, batch_size=32):
        while True:
            # Generate order of samples
            indices = np.arange(self.size)
            if shuffle:
                np.random.shuffle(indices)

            # Generate batches
            for start_id in range(0, self.size, batch_size):
                batch_ids = indices[start_id: start_id + batch_size]
                batch_images = self.images[batch_ids]
                batch_angles = self.inc_angle[batch_ids]
                batch_labels = self.labels[batch_ids]
                yield batch_images, batch_angles, batch_labels

