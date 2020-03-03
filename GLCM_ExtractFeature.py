import glob
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
import numpy as np


def GLCMFeatures(filepath):
    NORMALFINAL = []
    imageListNormal = []
    for item1 in glob.glob(filepath):
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
        # NormalStats3 = greycoprops(GLCMNORMAL, 'energy')
        Normal.append(NormalStats1[0][0])  # contrast
        Normal.append(NormalStats2[0][0])  # dissimilarity
        # Normal.append(NormalStats3[0][0])
        NORMALFINAL.append(Normal)
    return NORMALFINAL
