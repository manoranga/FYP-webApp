import numpy as np
import cv2
# import os
# import globS
# import pydicom as dicom
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from PIL import Image
import os, sys
import glob

imageList = []
segmentedImage = []
segmentedImage1 = []
segmentedLungMask = []
segmentedclosing = []
array1 = []
array2 = []


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


def grayscale_convert(image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] == 1 else 0

    # return the grayscale image
    return image


def segmentLung(imagefilepath):
    for item in glob.glob(imagefilepath):
        img = Image.open(item)
        imageList.append(img)

    len(imageList)
    for initial in imageList:
        dim = (1024, 1024)
        initial = initial.resize((1024, 1024))
        initial = np.array(initial).astype(np.uint8)
        gray1 = cv2.resize(initial, dim, interpolation=cv2.INTER_AREA)
        # gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.equalizeHist(initial)
        # median filter
        gray1 = cv2.medianBlur(gray1, 5)
        # plt.imshow( gray1)
        # noise removal
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        # plt.hist(gray1.ravel(), 256)

        # plt.show()
        # Threshold the image to binary using local method
        thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 699, 6)
        # plt.imshow( thresh1)
        constant = cv2.rectangle(thresh1, (0, 0), (1024, 1024), (255, 255, 255), 5)
        # finding cotours
        cnts, hierachy = cv2.findContours(constant, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)
        # print(constant.shape)
        draw_counters = initial.copy()
        cv2.drawContours(draw_counters, cnts, -1, (0, 255, 0), -1)
        # plt.imshow(draw_counters)
        # calculate number of pixel clusters
        ret, markers = cv2.connectedComponents(constant)
        # print(markers)
        label_hue = np.uint8(179 * markers / np.max(markers))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeled_img[label_hue == 0] = 0
        # plt.imshow(labeled_img)
        # Get the area taken by each component. Ignore label 0 since this is the background.
        marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
        # print(marker_area)
        # Get label of largest component by area
        sorted_marker_area = sorted(marker_area, reverse=True)
        # print(marker_area)
        largest_component_1 = marker_area.index(sorted_marker_area[1])
        largest_component_2 = marker_area.index(sorted_marker_area[2])
        # print(largest_component_1)
        # Get pixels which correspond to the lung
        lung_mask = markers == largest_component_1 + 1
        lung_mask = lung_mask + (markers == largest_component_2 + 1)
        # print(lung_mask)
        # lung_mask.astype(np.uint8)
        lung_mask = grayscale_convert(np.uint8(lung_mask))
        # print(lung_mask.shape)
        # plt.imshow(lung_mask)
        kernel = np.ones((32, 32), np.uint8)
        closing = cv2.morphologyEx(lung_mask, cv2.MORPH_DILATE, kernel)
        cv2.imwrite('static/SaveSegmentedLungOPenCV/Segmented_lung_openCV.jpeg', closing)

        segmentedLungMask.append(lung_mask)
        segmentedclosing.append(closing)

        a = []
        # area =0;
        for i in segmentedLungMask:
            area = 0;
            for x in i:
                for y in x:
                    if y == 255:
                        area += 1
            a.append(area)

            b = []
            for x in segmentedclosing:
                lung_perimeter = 0
                lung_perimeter = findperimeter(x)
                b.append(lung_perimeter)

            # import math
            # Equivalent_Diameter = []
            # Irregularity_Index = []
            # for IED in range(0, len(segmentedclosing), 1):
            #     I = 4 * 3.14 * a[IED] / b[IED] * b[IED]
            #     Irregularity_Index.append(I)
            #     ED = math.sqrt(4 * a[IED] / 3.14)
            #     Equivalent_Diameter.append(ED)
            # array1 = []
            # for final in range(0, len(segmentedclosing), 1):
            #     array2 = [b[final], Equivalent_Diameter[final]]
            #     array1.append(array2)

            # return array1
            return None

# print(segmentedclosing[0])
