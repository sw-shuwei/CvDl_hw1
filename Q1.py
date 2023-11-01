import os
import cv2
import glob
import numpy as np

class Question1:
    def __init__(self):
        self.word = None

    # HW1-1: Find Corners
    def find_corners(self):
        bmp_files = glob.glob('./Q1_Image/*.bmp') 
        bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        imgs = []
        for file in bmp_files:
            img = cv2.imread(file, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ret, corners = cv2.findChessboardCorners(img, (11, 8), cv2.CALIB_CB_ADAPTIVE_THRESH)
            img = cv2.drawChessboardCorners(img, (11, 8), corners, ret)
            imgs.append(img)

        imgs = np.array(imgs)
        for i in range(len(bmp_files)):
            if i>0:
                cv2.destroyWindow('Corner Detection-' + bmp_files[i-1])
            cv2.imshow('Corner Detection-' + bmp_files[i], imgs[i])
            cv2.waitKey(500)
            
        cv2.destroyAllWindows() 

    # HW1-2: find intrinsic matrix
    def find_intrinsic_matrix(self):
        bmp_files = glob.glob('./Q1_Image/*.bmp') 
        bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for file in bmp_files:
            img = cv2.imread(file, 0)
            ret, corners = cv2.findChessboardCorners(img, (11,8), cv2.CALIB_CB_ADAPTIVE_THRESH)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None) 
        print('intrinsic:\n', mtx)

    # HW1-3: find extrinsic matrix
    def find_extrinsic_matrix(self, n):
        bmp_file = os.path.join('Q1_Image', n+'.bmp')
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        img = cv2.imread(bmp_file, 0)
        ret, corners = cv2.findChessboardCorners(img, (11,8), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            Extrinsic_mx=[]
        
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
            Extrinsic_mx, jacobian= cv2.Rodrigues(np.float32(rvecs[int(n)-1]),np.float32(Extrinsic_mx)) 
                
            print('Extrinsic:\n', np.hstack((Extrinsic_mx, tvecs[int(n)-1])))

    # HW1-4: find distortion matrix
    def find_distortion_matrix(self):
        bmp_files = glob.glob('./Q1_Image/*.bmp') 
        bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for file in bmp_files:
            img = cv2.imread(file, 0)
            ret, corners = cv2.findChessboardCorners(img, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
        print('Distortion:\n', dist)

    # HW1-5: show undistorted result
    def show_undistorted_result(self):
        bmp_files = glob.glob('./Q1_Image/*.bmp') 
        bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for file in bmp_files:
            img = cv2.imread(file, 0)
            ret, corners = cv2.findChessboardCorners(img, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (11,8), 1, (11,8))
        imgs = []
        for file in bmp_files:
            img = cv2.imread(file, 0)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            # undistort
            #undistortion_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            undistortion_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)    

            # label text
            cv2.putText(img, 'Distored image', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(undistortion_img, 'Undistored image', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

            result = np.concatenate((img, undistortion_img), axis=1) 
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            imgs.append(result)
        
        imgs = np.array(imgs)

        for i in range(len(bmp_files)):
            if i>0:
                cv2.destroyWindow('Undistorted result ' + bmp_files[i])
            cv2.imshow('Undistorted result ' + bmp_files[i], imgs[i])
            cv2.waitKey(500)
        cv2.destroyAllWindows()

