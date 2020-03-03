import inline
import matplotlib
import numpy as np
import cv2

from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from PIL import Image
import os, sys
import glob

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

# calculate perimeter
# Python 3 program to find perimeter of area
# covered by 1 in 2D matrix consisits of 0's and 1's.

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


def getFeatures(filepath):
    for item in glob.glob(filepath):
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

        import math
        Equivalent_Diameter = []
        Irregularity_Index = []
        for IED in range(0, len(segmentedLungMask), 1):
            I = 4 * 3.14 * a[IED] / b[IED] * b[IED]
            Irregularity_Index.append(I)
            ED = math.sqrt(4 * a[IED] / 3.14)
            Equivalent_Diameter.append(ED)

        array2 = []
        array1 = []
        for final in range(0, len(Lung_Dis), 1):
            array2 = []
            array2.append(Lung_Dis[final])  # lungDistance
            array2.append(Dist_Throatic[final])  # distance of throactic
            array2.append(CTR[final])  # ratio of cardio trasics
            array2.append(LungAreaDefference[final])  # diffrence of lung area
            array2.append(a[final])  # lung area
            array2.append(b[final])  # lung perimeter
            array2.append(Irregularity_Index[final])  # Irregularity_Index
            array2.append(Equivalent_Diameter[final])  # Equivalent_Diameter
            array1.append(array2)

    return array1;


