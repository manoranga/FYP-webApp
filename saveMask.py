# From: https://github.com/zhixuhao/unet/blob/master/data.py
import cv2
import numpy as np
import os


def test_load_image_changed(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = img.astype(np.uint8)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator_changed(test_files, target_size=(256,256)):
    yield test_load_image_changed(test_files, target_size)


def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)
        print(img)
        return img

