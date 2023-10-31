import os
import cv2
import glob
import numpy as np


class Question4:
    def __init__(self):
        self.word = ''
    
    def keypoints(self):
        # Loading the image
        img = cv2.imread('Q4_image/Left.jpg')
        
        # Converting image to grayscale
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Applying SIFT detector
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        
        # Marking the keypoint on the image using circles
        img = cv2.drawKeypoints(gray, kp,img, color=(0,255,0))
        
        cv2.namedWindow('Keypoints', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Keypoints', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def matched_keypoints(self):
        # Loading the image
        img1 = cv2.imread('Q4_image/Left.jpg',cv2.IMREAD_GRAYSCALE)  # queryImage
        img2 = cv2.imread('Q4_image/Right.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
        
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.namedWindow('Matched Keypoints', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Matched Keypoints', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()