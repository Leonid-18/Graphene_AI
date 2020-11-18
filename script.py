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

class AnalysePhoto:
    
    def __init__(self, path):
        self.path = path
        
    def __read__(self, image):
        for img in glob.glob(self.path):
            print('PATH')
            image = cv2.imread(img)
        return image
    
    def __resize__(self, image):
        get_image = self.__read__(image)
        image_resize= get_image[0:1768, 0:2048]
        return image_resize
        
    def __toArray__(self):
        for img in glob.glob(self.path):
            image = cv2.imread(img)
            image_item = np.array(image)
            #image_array.append(image_item)
        return image_item
    
    def __gaussianBlur__(self, image):
        get_image = self.__resize__(image)
        gray = cv2.cvtColor(get_image,cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        return blur_gray
    
    def __cannyEdge__(self, gray):
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    def __autoCanny__(self,image, sigma):
        median = np.median(image)
        lower = int(max(0, (1.0 - sigma)*median))
        upper = int(min(255, (1.0 + sigma)*median))
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
         for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
          
        return cv2.addWeighted(img, 0.8, line_image, 1, 0)
        
        
        
        

image1 = AnalysePhoto('./Data/G038_002.tif')
image2 = AnalysePhoto('./Data/G038_002.tif')

gaussian = image1.__gaussianBlur__(image1)

edges = image1.__cannyEdge__(gaussian)

auto_edges = image2.__autoCanny__(gaussian, 0.3)

lines_edges = image1.__HoughLinesP__(image1.__resize__(image1), edges)

lines_auto_edges = image2.__HoughLinesP__(image2.__resize__(image1), auto_edges)



cv2.imshow("Gaussian image", gaussian)
cv2.imshow("Canny image", edges)
cv2.imshow("Auto Canny image", auto_edges)
cv2.imshow('HoughLinesP', lines_edges)
cv2.imshow('Auto HoughLinesP', lines_auto_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

#image2 = Image.fromarray(image1.__toArray__())
#print(image2)