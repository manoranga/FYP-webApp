import matplotlib.pyplot as plt
import os, sys
import glob
import array
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas as mt


def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

def extract_for_knn(imagefilepath_knn):
    PN = []
    PNFINAL = []
    imageListPN = []
    for item2 in glob.glob(imagefilepath_knn):
        img2 = Image.open(item2)
        imageListPN.append(img2)

    len(imageListPN)
    for initialPN in imageListPN:
        PN = []
        imPN = np.array(initialPN).astype(np.uint8)
        GLCMPN = greycomatrix(imPN, [1], [0], 256, symmetric=False, normed=True)
        PNStats1 = greycoprops(GLCMPN, 'contrast')
        PNStats2 = greycoprops(GLCMPN, 'dissimilarity')
        PNStats3 = greycoprops(GLCMPN, 'energy')
        PNHarlik = extract_features(imPN)
        PN.append(PNStats1[0][0])
        PN.append(PNStats2[0][0])
        PN.append(PNStats3[0][0])
        PN.append(PNHarlik[0])
        PN.append(PNHarlik[1])
        PNFINAL.append(PN)
    return PNFINAL;

def extract_for_randomForestt(imagefilepath_randomForest):
    Normal = []
    NORMALFINAL = []
    NORMALFINAL1 = []
    imageListNormal = []
    for item1 in glob.glob(imagefilepath_randomForest):
        img1 = Image.open(item1)
        imageListNormal.append(img1)

    len(imageListNormal)
    for initialNormal in imageListNormal:
        Normal = []
        # imNORMAL = cv2.cvtColor(initialNormal, cv2.COLOR_BGR2GRAY)
        imNORMAL = np.array(initialNormal).astype(np.uint8)
        GLCMNORMAL = greycomatrix(imNORMAL, [1], [0], 256, symmetric=False, normed=True)
        NormalStats1 = greycoprops(GLCMNORMAL, 'contrast')
        NormalStats2 = greycoprops(GLCMNORMAL, 'dissimilarity')
        NormalStats3 = greycoprops(GLCMNORMAL, 'energy')
        NormalHarlik = extract_features(imNORMAL)
        Normal.append(NormalStats1[0][0])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalStats2[0][0])
        Normal = []
        NORMALFINAL.append(Normal)
        Normal.append(NormalStats3[0][0])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalHarlik[0])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalHarlik[1])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalHarlik[2])
        NORMALFINAL.append(Normal)
    # extract morphological features

    #create matrix for random forest prediction
    matrix = [[0 for x in range(5)] for y in range(1)]
    matrix[0][0] = NORMALFINAL[0][0]
    matrix[0][1] = NORMALFINAL[1][0]
    matrix[0][2] = NORMALFINAL[2][0]
    matrix[0][3] = NORMALFINAL[3][0]
    matrix[0][4] = NORMALFINAL[4][0]

    return matrix;



