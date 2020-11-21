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

class AnalysePhoto:
    
    def __init__(self, path):
        self.path = path
        
    def __read__(self, image):
        for img in glob.glob(self.path):
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
    
    def __addBrightness__(self, image):
        
        # Converting image to LAB Color model
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Splitting the LAB image to different channels
        l, a, b = cv2.split(lab)
        
        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
        
    
    def __erode__(self, image, erode):
        element = np.ones((erode,erode), 'uint8')
        return cv2.erode(image, element, iterations=1)
    
    def __gaussianBlur__(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        return blur_gray
    
    
    def __cannyEdge__(self, gray, low_threshold, high_threshold):
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
        
        
        
        

image = AnalysePhoto('./Data/G038_002.tif')

image_first = image.__resize__(image)

#-----Gaussian-------------------------------------------
gaussian = image.__gaussianBlur__(image_first)

#-----Manual Cany -------------------------------------------
edges = image.__cannyEdge__(gaussian, 50, 150)
lines_edges = image.__HoughLinesP__(image_first, edges)

#-----Auto Cany-------------------------------------------

auto_edges = image.__autoCanny__(gaussian, 0.3)
lines_auto_edges = image.__HoughLinesP__(image_first, auto_edges)

#-----Gaussian with Erode-------------------------------------------

gaussina_erode = image.__erode__(gaussian, 2)

#-----Auto Cany with Erode-------------------------------------------

auto_edges_erode = image.__autoCanny__(gaussina_erode, 0.3)
lines_auto_edges_erode = image.__HoughLinesP__(image_first, auto_edges_erode)

#-----Manual Cany with Erode -------------------------------------------

manual_edges_erode = image.__cannyEdge__(gaussina_erode, 50, 150)
lines_edges_manual = image.__HoughLinesP__(image_first, manual_edges_erode)


#-----Gaussian with Erode and correcting brightness----------------------------


addBrightness = image.__addBrightness__(image_first)

gaussian_brightness = image.__gaussianBlur__(addBrightness)

gaussina_brightness_erode = image.__erode__(gaussian_brightness, 3)

manual_edges_erode_brightness = image.__cannyEdge__(gaussina_brightness_erode, 200, 300)
lines_edges_manual_brightness = image.__HoughLinesP__(image_first, manual_edges_erode_brightness)







cv2.imshow('Erode', gaussina_erode)
cv2.imshow("Gaussian image", gaussian)
cv2.imshow("Canny image", edges)

#cv2.imshow("Auto Canny image", auto_edges)
cv2.imshow('HoughLinesP', lines_edges)
#cv2.imshow('Auto HoughLinesP', lines_auto_edges)

cv2.imshow('HoughLinesP with Erode', lines_edges_manual)
#cv2.imshow('Auto HoughLinesP with Erode', lines_auto_edges_erode)
cv2.imshow('correcting brightness', addBrightness)
cv2.imshow('HoughLinesP with Erode and correcting brightness', lines_edges_manual_brightness)

hist,bins = np.histogram(lines_edges_manual.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(lines_edges_manual.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

#image2 = Image.fromarray(image1.__toArray__())
#print(image2)