#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:41:20 2020

@author: leonidtkachenko
"""

import numpy as np
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
import csv


class AnalysePhoto:

    def __init__(self, path):
        self.path = path

    def __read__(self, image):
        for img in glob.glob(self.path):
            image = cv2.imread(img)
        return image

    def __resize__(self, image):
        get_image = self.__read__(image)
        image_resize = get_image[0:1768, 0:2048]
        return image_resize

    def __toArray__(self):
        for img in glob.glob(self.path):
            image = cv2.imread(img)
            image_item = np.array(image)
            # image_array.append(image_item)
        return image_item

    def __addBrightness__(self, image):

        # Converting image to LAB Color model
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Splitting the LAB image to different channels
        l, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(11, 11))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def __erode__(self, image, erode):
        element = np.ones((erode, erode), 'uint8')
        return cv2.erode(image, element, iterations=1)

    def __gaussianBlur__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        return blur_gray

    def __cannyEdge__(self, gray, low_threshold, high_threshold):
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def __autoCanny__(self, image, sigma):
        median = np.median(image)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        auto_edges = cv2.Canny(image, lower, upper)
        return auto_edges

    def __HoughLinesP__(self, img, edges):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 15  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
        lines_on_original_photo = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        return lines_on_original_photo, line_image

    def __histogram__(self, img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.plot(cdf_normalized, color='b')
        plt.hist(img.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        return plt.gcf()

    def __find_contors__(self, img):
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15, 15)))
        cv2.imshow("closed_img", closing)

    def getNeighbours(self, i, j, n, m):
        arr = []
        if i - 1 >= 0 and j - 1 >= 0:
            arr.append((i - 1, j - 1))
        if i - 1 >= 0:
            arr.append((i - 1, j))
        if i - 1 >= 0 and j + 1 < m:
            arr.append((i - 1, j + 1))
        if j + 1 < m:
            arr.append((i, j + 1))
        if i + 1 < n and j + 1 < m:
            arr.append((i + 1, j + 1))
        if i + 1 < n:
            arr.append((i + 1, j))
        if i + 1 < n and j - 1 >= 0:
            arr.append((i + 1, j - 1))
        if j - 1 >= 0:
            arr.append((i, j - 1))
        return arr

    def __floodFill__(self, img):
        h, w = img.shape[:2]
        diff = (6, 6, 6)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        nelem = 0
        waste = 0
        i = 0
        j = 0
        g = 0
        arr_rect = []
        area = []
        all_area = []
        for x in range(1767):
            for y in range(2047):
                if img[x, y].any() == 0:
                    if nelem == 0 or nelem % 10 == 0:
                        i = 50
                        j = 144
                        g = 5
                    if nelem == 1 or nelem % 10 == 1:
                        i = 230
                        j = 80
                        g = 100
                    if nelem == 2 or nelem % 10 == 2:
                        i = 78
                        j = 3
                        g = 240
                    if nelem == 3 or nelem % 10 == 3:
                        i = 3
                        j = 250
                        g = 200
                    if nelem == 4 or nelem % 10 == 4:
                        i = 170
                        j = 39
                        g = 150
                    if nelem == 5 or nelem % 10 == 5:
                        i = 100
                        j = 240
                        g = 5
                    if nelem == 6 or nelem % 10 == 6:
                        i = 32
                        j = 240
                        g = 200
                    if nelem == 7 or nelem % 10 == 7:
                        i = 176
                        j = 134
                        g = 13
                    if nelem == 8 or nelem % 10 == 8:
                        i = 45
                        j = 45
                        g = 120
                    if nelem == 9 or nelem % 10 == 9:
                        i = 3
                        j = 158
                        g = 200
                    nelem += 1

                    retval, image, mask, rect = cv2.floodFill(img, mask, (y, x), (g, i, j), diff, diff)
                    all_area.append(retval)
                    if retval < 20:  # 0 - x, 1 -y, 2 - length x , 3 - length y
                        cv2.floodFill(img, None, (y, x), (255, 255, 255), diff, diff)
                        waste += 1
                    else:
                        arr_rect.append(rect)
                        area.append(retval)  # retval - area (count of pixels)
        with  open('test.txt', 'a') as f:
            f.write(str(arr_rect))
            f.write('\n' + str(area))
            f.write('\n' + str(len(arr_rect)))
        print('Element', nelem)
        print('Waste', waste)
        print('Number islands', nelem - waste)
        # print('Area', area)
        print(i, g, i)
        # n, bins, patches = plt.hist(x=area, bins='auto', color='#0504aa',
        #                             alpha=0.7, rwidth=0.85)
        # plt.grid(axis='y', alpha=0.75)
        # plt.xlabel('Islands Area, pixels')
        # plt.ylabel('Quantity')
        # plt.title('h-BN')
        # Set a clean upper y-axis limit.
        # plt.ylim(ymax=960)
        # plt.xlim(xmax=700, xmin=0)
        # plt.show()
        # cv2.imshow('hh', img)

        return img, arr_rect, area, all_area

    def kmeans_color_quantization(self, image, clusters=8, rounds=1):
        h, w = image.shape[:2]
        samples = np.zeros([h * w, 3], dtype=np.float32)
        count = 0

        for x in range(h):
            for y in range(w):
                samples[count] = image[x][y]
                count += 1

        compactness, labels, centers = cv2.kmeans(samples,
                                                  clusters,
                                                  None,
                                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                                  rounds,
                                                  cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape((image.shape))

    def __create_csv__(self, arr_rect, area, all_area):
        with open('analysis.csv', 'w', newline='') as f:
            thewriter = csv.writer(f)
            all_island_area = 0
            thewriter.writerow(['x','y','Area per island','Quantity','All islands area, pixels', 'Line area'])
            for i in range(len(area)):
                all_island_area+=area[i]
            for i in range(len(area)):
                if i == 0:
                 thewriter.writerow([arr_rect[i][0],arr_rect[i][1],area[i],len(area),
                                    all_island_area,2048*1768-all_island_area])
                else:
                    thewriter.writerow([arr_rect[i][0],arr_rect[i][1],area[i]])



image = AnalysePhoto('./Data/G038_002.tif')

image_first = image.__resize__(image)

# -----Gaussian-------------------------------------------
gaussian = image.__gaussianBlur__(image_first)

# -----Manual Cany -------------------------------------------
edges = image.__cannyEdge__(gaussian, 50, 150)
lines_edges = image.__HoughLinesP__(image_first, edges)

# -----Auto Cany-------------------------------------------

auto_edges = image.__autoCanny__(gaussian, 0.3)
lines_auto_edges = image.__HoughLinesP__(image_first, auto_edges)

# -----Gaussian with Erode-------------------------------------------

gaussina_erode = image.__erode__(gaussian, 2)

# -----Auto Cany with Erode-------------------------------------------

auto_edges_erode = image.__autoCanny__(gaussina_erode, 0.3)
lines_auto_edges_erode = image.__HoughLinesP__(image_first, auto_edges_erode)

# -----Manual Cany with Erode ----------------------------------------

manual_edges_erode = image.__cannyEdge__(gaussina_erode, 50, 150)
# lines_edges_manual = image.__HoughLinesP__(image_first, manual_edges_erode)

# -----Gaussian with Erode and correcting brightness------------------


addBrightness = image.__addBrightness__(image_first)
gaussian_brightness = image.__gaussianBlur__(addBrightness)
gaussina_brightness_erode = image.__erode__(gaussian_brightness, 3)

# -----Manual Cany with Erode and correcting brightness------------------

manual_edges_erode_brightness = image.__cannyEdge__(gaussina_brightness_erode, 200, 300)
lines_edges_manual_brightness, dark_photo = image.__HoughLinesP__(image_first, manual_edges_erode_brightness)

# -----Histogram------------------
# primary_image = image.__histogram__(image_first)
# plt.show()
# primary_image_with_cvt = image.__histogram__(addBrightness)
# plt.show()
# processed_image = image.__histogram__(lines_edges_manual_brightness)

# -----Draw-Contors-----------------
image2 = AnalysePhoto('HoughLinesP_Black.png')
image_sec = image2.__resize__(image_first)
# image2.__find_contors__(image_sec)

# --------FloodFill-----------------------
color_island, arr_rect, area, all_area = image2.__floodFill__(image_sec)
# lines_edges_manual_brightness_d = image.__HoughLinesP__(image_sec, manual_edges_erode_brightness_d)
# cv2.imshow('err', lines_edges_manual_brightness_d)
# ---------To_CVS_FILE--------------------
image2.__create_csv__(arr_rect, area, all_area)
# kmeans = image2.kmeans_color_quantization(image_sec, clusters=3)
# result = kmeans.copy()

# # Floodfill
# seed_point = (150, 50)
# cv2.floodFill(result, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))

# cv2.imshow('kmeans', kmeans)
# cv2.imshow('result', result)


# -----Save-image------------------
# primary_image.savefig("Histogram_of_primary_image.png")
# primary_image_with_cvt.savefig("Histogram_of_primary_image_with_cvt.png")
# processed_image.savefig("Histogram_of_processed_image.png")
Image.fromarray(lines_edges_manual_brightness).save("HoughLinesP_with_Manual_Cany_Erode_Correcting_brightness.png")
Image.fromarray(dark_photo).save("HoughLinesP_Black.png")
Image.fromarray(color_island).save("HoughLinesP_Colored.png")
# Image.fromarray(lines_edges_manual).save("HoughLinesP_with_Manual_Cany_Erode.png")
# Image.fromarray(lines_edges).save("HoughLinesP_with_Manual_Cany.png")

# cv2.imshow('Erode', gaussina_erode)
# cv2.imshow("Gaussian image", gaussian)
# cv2.imshow("Canny image", edges)

# cv2.imshow("Auto Canny image", auto_edges)
# cv2.imshow('HoughLinesP', lines_edges)
# cv2.imshow('Auto HoughLinesP', lines_auto_edges)

# cv2.imshow('HoughLinesP with Erode', lines_edges_manual)
# cv2.imshow('Auto HoughLinesP with Erode', lines_auto_edges_erode)
# cv2.imshow('correcting brightness', addBrightness)
# cv2.imshow('HoughLinesP with Erode and correcting brightness', lines_edges_manual_brightness)

cv2.waitKey(0)
cv2.destroyAllWindows()
