import cv2
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from models import unet_model
from tqdm import tqdm
from numba import prange, njit
import pandas as pd
import os
from utils import abs_path
from glob import glob
from matplotlib import pyplot as plt
import mahotas as mt

class Image:
    def __init__(self, image_file, divide=False, reshape=False):
        self.image_file = image_file
        self.divide = divide
        self.reshape = reshape
        self.data = self.__load_file()

    def __load_file(self, target_size=(512, 512)):
        img = cv2.imread(self.image_file, cv2.IMREAD_GRAYSCALE)
        if self.divide:
            img = img / 255
        img = cv2.resize(img, target_size)
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
        return img

    def get_file_dir(self):
        return os.path.splitext(os.path.basename(self.image_file))

    def save_to(self, path_dir):
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            path_dir, "%s_processed%s" % (filename, fileext))
        cv2.imwrite(result_file, self.data)

    def shape(self):
        return self.data.shape

    def hist(self):
        result = np.squeeze(cv2.calcHist(
            [self.data], [0], None, [255], [1, 256]))
        result = np.asarray(result, dtype='int32')
        return result

    def save_hist(self, save_folder=''):
        plt.figure()
        histg = cv2.calcHist([self.data], [0], None, [254], [
            1, 255])  # calculating histogram
        plt.plot(histg)
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            save_folder, "%s_histogram%s" % (filename, '.png'))
        plt.savefig(result_file)
        plt.close()

    def haralick(self):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


class ImageGenerator:
    def generate_from(self, path, divide=False, reshape=False, only_data=False):
        image_files = glob(path + "/*g")
        for image_file in image_files:
            if only_data:
                yield Image(image_file, divide, reshape).data
            else:
                yield Image(image_file, divide, reshape)

    def generate_preprocessing_data(self,
                                    covid_path,
                                    covid_masks_path,
                                    non_covid_path,
                                    non_covid_masks_path):
        with ThreadPoolExecutor() as executor:
            covid_images = executor.submit(self.generate_from, covid_path)
            covid_masks = executor.submit(self.generate_from, covid_masks_path)

            non_covid_images = executor.submit(
                self.generate_from, non_covid_path)
            non_covid_masks = executor.submit(
                self.generate_from, non_covid_masks_path)

            return [covid_images, covid_masks, non_covid_images, non_covid_masks]

    def generate_classificator_data(self, covid_path, non_covid_path, divide=True, reshape=False):
        with ThreadPoolExecutor() as executor:
            covid_images = executor.submit(
                self.generate_from, covid_path, divide, reshape, True)

            non_covid_images = executor.submit(
                self.generate_from, non_covid_path, divide, reshape, True)

            covid_images = list(covid_images.result())
            non_covid_images = list(non_covid_images.result())

            entries = np.concatenate((covid_images, non_covid_images))
            entries = np.repeat(entries[..., np.newaxis], 3, -1)

            cov_len = len(covid_images)
            non_cov_len = len(non_covid_images)
            results_len = cov_len + non_cov_len
            results = np.zeros((results_len))

            results[0:cov_len] = 1

            # Split into test and training
            return train_test_split(
                entries, results, test_size=0.2, random_state=0)

    def generate_processed_data(self, covid_processed_path, non_covid_processed_path, divide=True, reshape=False):
        with ThreadPoolExecutor() as executor:
            covid_images = executor.submit(
                self.generate_from, covid_processed_path)
            non_covid_images = executor.submit(
                self.generate_from, non_covid_processed_path)

            return [covid_images, non_covid_images]


class ImageSaver:
    def __init__(self, images):
        self.images = images

    def save_to(self, path_dir):
        for img in self.images:
            img.save_to(path_dir)


class ImageProcessor:

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    @staticmethod
    @njit()
    def __apply_mask(img, mask):
        imshape = img.shape

        for i in prange(0, imshape[0]):  # imshape[0] = 512
            for j in prange(0, imshape[1]):  # imshape[1] = 512
                if mask[i][j] <= 20:
                    img[i][j] = 0
        return img

    def __process_image(self, args):
        img, mask = args
        img.data = cv2.equalizeHist(img.data)
        img.data = ImageProcessor.__apply_mask(
            np.asarray(img.data), np.asarray(mask.data))
        return img

    def process(self):
        iterables = np.swapaxes([self.images, self.masks], 0, 1)
        total_size = len(iterables)
        list = []
        for i in tqdm(prange(total_size)):
            list.append(self.__process_image(iterables[i]))
        return list


class ImageSegmentator:
    """
    Code using the model of the work accessed in:
    https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/execution#4.-Results
    """

    def __init__(self, input_size=(512, 512, 1),
                 target_size=(512, 512),
                 folder_in='',
                 folder_out=''):
        self.input_size = input_size
        self.target_size = target_size
        self.folder_in = folder_in
        self.folder_out = folder_out

    def __test_load_image(self, test_file):
        img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, self.target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        return img

    def __test_generator(self, test_files):
        for test_file in test_files:
            yield self.__test_load_image(test_file)

    def __save_result(self, save_path, npyfile, test_files):
        for i, item in enumerate(npyfile):
            result_file = test_files[i]
            img = (item[:, :, 0] * 255.).astype(np.uint8)

            filename, fileext = os.path.splitext(os.path.basename(result_file))

            result_file = abs_path(
                save_path, "%s_predict%s" % (filename, fileext))

            cv2.imwrite(result_file, img)

    def segmentate(self):
        model = unet_model(input_size=self.input_size)
        model.load_weights('segmentation_model.hdf5')

        test_files = glob(self.folder_in + "/*g")

        test_gen = self.__test_generator(
            test_files)
        results = model.predict_generator(test_gen, len(test_files), verbose=1)
        self.__save_result(self.folder_out, results, test_files)


class ImageCharacteristics:
    def __init__(self, cov_images, non_cov_images):
        self.cov_images = cov_images
        self.non_cov_images = non_cov_images

    def save(self, file_path):
        #Histogram
        data = [np.hstack((img.hist(), img.haralick(), [0])) for img in self.non_cov_images]
        data += [np.hstack((img.hist(), img.haralick(), [1])) for img in self.cov_images]

        pd.DataFrame(data).to_csv(file_path)