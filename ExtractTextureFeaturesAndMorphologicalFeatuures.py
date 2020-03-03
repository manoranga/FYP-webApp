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

imageList =[]
segmentedImage =[]
segmentedImage1 =[]
segmentedLungMask=[]
segmentedclosing =[]
array1=[]
array2=[]
CTR =[]
Dist_Throatic=[]
Lung_Dis =[]
FinalPN =[]
FinalPNF1 =[]
LungAreaDefference =[]

# forKNN
imageList1 =[]
segmentedImage1 =[]
segmentedImage11 =[]
segmentedLungMask1=[]
segmentedclosing1 =[]
array11=[]
array21=[]
CTR1 =[]
Dist_Throatic1=[]
Lung_Dis1 =[]
FinalPN1 =[]
FinalPNF11 =[]
LungAreaDefference1 =[]

R = 1024
C = 1024


# Find the number of covered side for mat[i][j].
def numofneighbour(mat, i, j):
    count = 0;

    # UP
    if i > 0 and mat[i - 1][j]:
        count += 1;

    # LEFT
    if j > 0 and mat[i][j - 1]:
        count += 1;

    # DOWN
    if i < R - 1 and mat[i + 1][j]:
        count += 1

    # RIGHT
    if j < C - 1 and mat[i][j + 1]:
        count += 1;

    return count;


# Returns sum of perimeter of shapes formed with 1s
def findperimeter(mat):
    perimeter = 0;

    # Traversing the matrix and finding ones to
    # calculate their contribution.
    for i in range(0, R):
        for j in range(0, C):
            if mat[i][j]:
                perimeter += (4 - numofneighbour(mat, i, j));

    return perimeter;


def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

def extract_for_knn(imagefilepath_knn,maskImagepathforKNN):
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
        # PNFINAL.append(PN)

        #extract morphological features

    for item in glob.glob(maskImagepathforKNN):
        img = Image.open(item)
        imageList1.append(img)

    len(imageList1)
    for initial in imageList1:
        dim = (1024, 1024)
        initial = initial.resize((1024, 1024))
        initial = np.array(initial).astype(np.uint8)
        lung_mask = cv2.resize(initial, dim, interpolation=cv2.INTER_AREA)

        #############################################################################################################################
        import array
        arr_i = array.array('i')
        arr_j = array.array('i')

        # get the right most pixel
        for i in range(0, 250, 1):  # along x axis
            for j in range(0, 1024, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_i.append(j)
                    arr_j.append(i)
                    arr_rm = [arr_i[0], arr_j[0]]

        arr_x = array.array('i')
        arr_y = array.array('i')

        # get the left most pixel
        for i in range(1023, 750, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_x.append(j)
                    arr_y.append(i)
                    arr_lm = [arr_x[0], arr_y[0]]

        mp_x = (arr_rm[0] + arr_lm[0]) / 2
        mp_y = (arr_rm[1] + arr_lm[1]) / 2

        # right lung area
        area_r = 0
        mp_y = int(mp_y)

        for i in range(mp_y, 1023, 1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    area_r = area_r + 1

        # left lung area
        area_l = 0
        mp_y = int(mp_y)

        for i in range(mp_y, 0, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    area_l = area_l + 1

        # differance of areas  and appe LungAreaDefference
        diff = 0
        diff = area_l - area_r
        LungAreaDefference1.append(diff)

        dist_lung = arr_lm[1] - arr_rm[1]
        Lung_Dis1.append(dist_lung)

        # get right throatic point
        arr_a = []
        arr_b = []
        mp_y = int(mp_y)

        for i in range(mp_y, 750, 1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_a.append(j)
                    arr_b.append(i)

                    arr_mr = [arr_a[0], arr_b[0]]

        # get left throatic point

        arr_c = []
        arr_d = []
        mp_y = int(mp_y)

        for i in range(mp_y, 250, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_c.append(j)
                    arr_d.append(i)

                    arr_ml = [arr_c[0], arr_d[0]]

        # caculate thotic distance and append array

        dist_throatic = arr_mr[1] - arr_ml[1]
        Dist_Throatic1.append(dist_throatic)

        # Calculate CTR
        ratio = dist_throatic / dist_lung

        CTR1.append(ratio)
        segmentedLungMask1.append(lung_mask)

        b1 = []
        for x in segmentedLungMask1:
            lung_perimeter = 0
            lung_perimeter = findperimeter(x)
            b1.append(lung_perimeter)

        a1 = []
        # area =0;
        for i in segmentedLungMask1:
            area = 0;
            for x in i:
                for y in x:
                    if (y == 255):
                        area += 1
            a1.append(area)

        # import math
        # Equivalent_Diameter1 = []
        # Irregularity_Index1 = []
        # for IED in range(0, len(segmentedLungMask1), 1):
        #     I = 4 * 3.14 * a1[IED] / b1[IED] * b1[IED]
        #     Irregularity_Index1.append(I)
        #     ED = math.sqrt(4 * a1[IED] / 3.14)
        #     Equivalent_Diameter1.append(ED)

        array2 = []
        array1 = []
        for final in range(0, len(Lung_Dis1), 1):
            array2 = []
            PN.append(Lung_Dis1[final])
            PN.append(CTR1[final])
            PN.append(LungAreaDefference1[final])
            PN.append(a1[final])
            PN.append(b1[final])
            PNFINAL.append(PN)
            # Normal = []
            # Normal.append(Lung_Dis[final])
            # NORMALFINAL.append(Normal)
            # Normal = []
            # Normal.append(CTR[final])
            # NORMALFINAL.append(Normal)
            # Normal = []
            # Normal.append(LungAreaDefference[final])
            # NORMALFINAL.append(Normal)
            # Normal = []
            # Normal.append(a[final])
            # NORMALFINAL.append(Normal)
            # Normal = []
            # Normal.append(b[final])
            # NORMALFINAL.append(Normal)

    return PNFINAL;

def extract_for_randomForestt(imagefilepath_randomForest,maskimagepath):
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
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalStats3[0][0])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalHarlik[0])
        NORMALFINAL.append(Normal)
        Normal = []
        Normal.append(NormalHarlik[1])
        NORMALFINAL.append(Normal)
        # Normal = []
        # Normal.append(NormalHarlik[2])
        # NORMALFINAL.append(Normal)


    # extract morphological features
    for item in glob.glob(maskimagepath):
        img = Image.open(item)
        imageList.append(img)

    len(imageList)
    for initial in imageList:
        dim = (1024, 1024)
        initial = initial.resize((1024, 1024))
        initial = np.array(initial).astype(np.uint8)
        lung_mask = cv2.resize(initial, dim, interpolation=cv2.INTER_AREA)

        #############################################################################################################################
        import array
        arr_i = array.array('i')
        arr_j = array.array('i')

        # get the right most pixel
        for i in range(0, 250, 1):  # along x axis
            for j in range(0, 1024, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_i.append(j)
                    arr_j.append(i)
                    arr_rm = [arr_i[0], arr_j[0]]

        arr_x = array.array('i')
        arr_y = array.array('i')

        # get the left most pixel
        for i in range(1023, 750, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_x.append(j)
                    arr_y.append(i)
                    arr_lm = [arr_x[0], arr_y[0]]

        mp_x = (arr_rm[0] + arr_lm[0]) / 2
        mp_y = (arr_rm[1] + arr_lm[1]) / 2

        # right lung area
        area_r = 0
        mp_y = int(mp_y)

        for i in range(mp_y, 1023, 1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    area_r = area_r + 1

        # left lung area
        area_l = 0
        mp_y = int(mp_y)

        for i in range(mp_y, 0, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    area_l = area_l + 1

        # differance of areas  and appe LungAreaDefference
        diff = 0
        diff = area_l - area_r
        LungAreaDefference.append(diff)

        dist_lung = arr_lm[1] - arr_rm[1]
        Lung_Dis.append(dist_lung)

        # get right throatic point
        arr_a = []
        arr_b = []
        mp_y = int(mp_y)

        for i in range(mp_y, 750, 1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_a.append(j)
                    arr_b.append(i)

                    arr_mr = [arr_a[0], arr_b[0]]

        # get left throatic point

        arr_c = []
        arr_d = []
        mp_y = int(mp_y)

        for i in range(mp_y, 250, -1):  # along x axis
            for j in range(0, 1023, 1):  # along the y axis
                if (lung_mask[j][i] == 255):  # all the white pixels
                    arr_c.append(j)
                    arr_d.append(i)

                    arr_ml = [arr_c[0], arr_d[0]]

        # caculate thotic distance and append array

        dist_throatic = arr_mr[1] - arr_ml[1]
        Dist_Throatic.append(dist_throatic)

        # Calculate CTR
        ratio = dist_throatic / dist_lung

        CTR.append(ratio)
        segmentedLungMask.append(lung_mask)

        b = []
        for x in segmentedLungMask:
            lung_perimeter = 0
            lung_perimeter = findperimeter(x)
            b.append(lung_perimeter)

        a = []
        # area =0;
        for i in segmentedLungMask:
            area = 0;
            for x in i:
                for y in x:
                    if (y == 255):
                        area += 1
            a.append(area)

        # import math
        # Equivalent_Diameter = []
        # Irregularity_Index = []
        # for IED in range(0, len(segmentedLungMask), 1):
        #     I = 4 * 3.14 * a[IED] / b[IED] * b[IED]
        #     Irregularity_Index.append(I)
        #     ED = math.sqrt(4 * a[IED] / 3.14)
        #     Equivalent_Diameter.append(ED)

        array2 = []
        array1 = []
        for final in range(0, len(Lung_Dis), 1):
            array2 = []
            Normal = []
            Normal.append(Lung_Dis[final])
            NORMALFINAL.append(Normal)
            Normal = []
            Normal.append(CTR[final])
            NORMALFINAL.append(Normal)
            Normal = []
            Normal.append(LungAreaDefference[final])
            NORMALFINAL.append(Normal)
            Normal = []
            Normal.append(a[final])
            NORMALFINAL.append(Normal)
            Normal = []
            Normal.append(b[final])
            NORMALFINAL.append(Normal)


            # array2.append(Lung_Dis[final])  # lungDistance
            # array2.append(Dist_Throatic[final])  # distance of throactic
            # array2.append(CTR[final])  # ratio of cardio trasics
            # array2.append(LungAreaDefference[final])  # diffrence of lung area
            # array2.append(a[final])  # lung area
            # array2.append(b[final])  # lung perimeter
            # array2.append(Irregularity_Index[final])  # Irregularity_Index
            # array2.append(Equivalent_Diameter[final])  # Equivalent_Diameter
            # array1.append(array2)



            #create matrix for random forest prediction
            matrix = [[0 for x in range(10)] for y in range(1)]
            matrix[0][0] = NORMALFINAL[0][0]
            matrix[0][1] = NORMALFINAL[1][0]
            matrix[0][2] = NORMALFINAL[2][0]
            matrix[0][3] = NORMALFINAL[3][0]
            matrix[0][4] = NORMALFINAL[4][0]
            matrix[0][5] = NORMALFINAL[5][0]
            matrix[0][6] = NORMALFINAL[6][0]
            matrix[0][7] = NORMALFINAL[7][0]
            matrix[0][8] = NORMALFINAL[8][0]
            matrix[0][9] = NORMALFINAL[9][0]

    return matrix;



