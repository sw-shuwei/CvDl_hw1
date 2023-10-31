import os
import cv2
import glob
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt

# from keras.datasets import cifar10

# import tensorflow
# import keras
# import torch
# import torchvision.models as models
# from pytorch_model_summary import summary

# import pytorch_model_summary as pms
w=' [wait for ESC key to exit]'

# def close_windows():
#     K=cv2.waitKey(0)
#     if K == 27: # wait for ESC key to exit
#         cv2.destroyAllWindows()

# def image_show(name, img):
#     plt.imshow(img)
#     plt.title(name)
#     plt.show()
#     plt.close()

# # HW1-1: Find Corners
# def find_corners(self):
#     bmp_files = glob.glob('./Q1_Image/*.bmp') 
#     bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
#     imgs = []
#     for file in bmp_files:
#         img = cv2.imread(file, 0)
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         ret, corners = cv2.findChessboardCorners(img, (11, 8), cv2.CALIB_CB_ADAPTIVE_THRESH)
#         img = cv2.drawChessboardCorners(img, (11, 8), corners, ret)
#         imgs.append(img)

#     imgs = np.array(imgs)
#     for i in range(len(bmp_files)):
#         plt.imshow(imgs[i])
#         plt.show(block=False)
#         plt.pause(0.6)  # sec
#         plt.close()

#     cv2.destroyAllWindows() 

# # HW1-2: find intrinsic matrix
# def find_intrinsic_matrix():
#     bmp_files = glob.glob('./Q1_Image/*.bmp') 
#     bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((8*11,3), np.float32)
#     objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     for file in bmp_files:
#         img = cv2.imread(file, 0)
#         ret, corners = cv2.findChessboardCorners(img, (11,8), cv2.CALIB_CB_ADAPTIVE_THRESH)
#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#             imgpoints.append(corners)
    
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None) 
#     print('intrinsic:\n', mtx)

# # HW1-3: find extrinsic matrix
# def find_extrinsic_matrix(n):
#     bmp_file = os.path.join('Q1_Image', n+'.bmp')
    
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((8*11,3), np.float32)
#     objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     img = cv2.imread(bmp_file, 0)
#     ret, corners = cv2.findChessboardCorners(img, (11,8), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         Extrinsic_mx=[]
    
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
#         Extrinsic_mx, jacobian= cv2.Rodrigues(np.float32(rvecs[int(n)-1]),np.float32(Extrinsic_mx)) 
            
#         print('Extrinsic:\n', np.hstack((Extrinsic_mx, tvecs[int(n)-1])))

# # HW1-4: find distortion matrix
# def find_distortion_matrix():
#     bmp_files = glob.glob('./Q1_Image/*.bmp') 
#     bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((8*11,3), np.float32)
#     objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     for file in bmp_files:
#         img = cv2.imread(file, 0)
#         ret, corners = cv2.findChessboardCorners(img, (11,8), None)
#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#             imgpoints.append(corners)
        
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
#     print('Distortion:\n', dist)

# # HW1-5: show undistorted result
# def show_undistorted_result():
#     bmp_files = glob.glob('./Q1_Image/*.bmp') 
#     bmp_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((8*11,3), np.float32)
#     objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     for file in bmp_files:
#         img = cv2.imread(file, 0)
#         ret, corners = cv2.findChessboardCorners(img, (11,8), None)
#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#             imgpoints.append(corners)

#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (11,8), 1, (11,8))
#     imgs = []
#     for file in bmp_files:
#         img = cv2.imread(file, 0)
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
#         h,  w = img.shape[:2]
#         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#         # undistort
#         #undistortion_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

#         # undistort
#         mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
#         undistortion_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)    

#         # label text
#         cv2.putText(img, 'Distored image', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
#         cv2.putText(undistortion_img, 'Undistored image', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

#         result = np.concatenate((img, undistortion_img), axis=1) 
#         result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
#         imgs.append(result)
    
#     imgs = np.array(imgs)

#     for i in range(len(bmp_files)):
#         plt.imshow(imgs[i])
#         plt.show(block=False)
#         plt.pause(0.6)  # sec
#         plt.close()

#     cv2.destroyAllWindows() 

# # HW2-1: show words on board
# def show_words_on_board(text):
#     fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
#     chessboard_AR(fs, text)

# # HW2-2: show words vertically on board
# def show_words_vertically(text):
#     fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
#     chessboard_AR(fs, text)

# def show_words(text, img, lines, rvecs, tvecs, mtx, dist):
#     # Space for showing words
#     space = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]])

#     # Shift the word to the Space
#     shift = np.ndarray.tolist(np.zeros(len(text)))
#     for i in range(len(lines)):
#         shift[i] = np.float32(lines[i] + space[i]).reshape(-1, 3)
#     shift = np.concatenate(tuple(shift), axis=0)

#     # project 3D points to image plane
#     imgpts, jac = cv2.projectPoints(shift, rvecs, tvecs, mtx, dist)

#     imgpts = imgpts.reshape(int(imgpts.shape[0] / 2), 2, 2).astype(int)

#     # draw lines
#     for l in range(len(imgpts)):
#         img = cv2.line(img, tuple(imgpts[l][0]), tuple(imgpts[l][1]), (255, 0, 0), 2)

#     return img

# def chessboard_AR(fs, text):
#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     objp = np.zeros((11 * 8, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpoints = []  # 2d points in image plane.

#     images = glob.glob('Q2_Image/*.bmp')
#     for fname in images:
#         img = cv2.imread(fname)
#         img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # Find the chess board corners
#         ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             imgpoints.append(corners2)

#             ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#             # Find the rotation and translation vectors.
#             _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

#             # get data from library
#             alphabet = np.ndarray.tolist(np.zeros(len(text)))
#             for w in range(len(text)):
#                 alphabet[w] = fs.getNode(text[w]).mat()

#             # Show word on chessboard
#             img = show_words(text, img, alphabet, rvecs, tvecs, mtx, dist)

#             plt.imshow(img)
#             plt.title('AR')
#             plt.show(block=False)
#             plt.pause(1)  # sec
#             plt.close()

#         cv2.destroyAllWindows() 

# 3-1: 
def stereo_display_map_btn():
    imgL = cv2.imread('Q4_Image/Left.jpg')
    cv2.namedWindow('imageL',cv2.WINDOW_NORMAL)
    cv2.resizeWindow("imgL", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
    cv2.setMouseCallback('imgL', self.draw_circle)
    cv2.imshow('imgL',imgL)

    cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
    cv2.resizeWindow("disparity", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
    cv2.imshow('disparity', self.disparity)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circle(self, event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
    
        img = cv2.cvtColor(np.copy(self.disparity),cv2.COLOR_GRAY2BGR)
        img_dot = cv2.cvtColor(np.copy(self.disparity) ,cv2.COLOR_GRAY2BGR)
        cv2.circle(img_dot,(x,y),10,(255,0,0),-1)

        mouseX,mouseY = x,y
        depth=baseline*focal_length/(img[y][x][0]+doffs)
        print(x,y)
        print(depth)

        imgR_dot = cv2.imread('Q4/imR.png')
        z=img[y][x][0]
        
        if img[y][x][0] != 0:       
            cv2.circle(imgR_dot,(x-z,y),25,(0,255,0),-1)
        
        cv2.namedWindow('imgR_dot',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgR_dot", int(imgR_dot.shape[1]/4), int(imgR_dot.shape[0]/4))
        cv2.imshow('imgR_dot', imgR_dot)           
        cv2.waitKey(0)


# 4.1 Resize = (256,256)
def hw4_1():
    # global square
    # square=cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
    
    # cv2.imshow('Resize = (256,256)'+w,square)
    # close_windows()
    pass



# 4.2 Translation
def hw4_2():
    # global square

    # M = np.float32([[1,0,0],[0,1,60]]) # X=0 Y=60
    # square = cv2.warpAffine(square,M,(400,300))
    
    # cv2.imshow('Translation'+w,square)
    # close_windows()
    pass


# 4.3 Rotation & Scaling
def hw4_3():
    # global square
    # # center:旋轉中心，angle:旋轉角度，scale:縮放比例, 生成一２＊３的矩陣
    # M = cv2.getRotationMatrix2D((128,188),10,0.5)
    # square = cv2.warpAffine(square,M,(400,300))

    # cv2.imshow('Rotation'+w,square)
    # close_windows()
    pass


# 4.4 Shearing
def hw4_4():
    # global square

    # pts1 = np.float32([[50,50],[200,50],[50,200]])
    # pts2 = np.float32([[10,100],[200,50],[100,250]])

    # M = cv2.getAffineTransform(pts1,pts2)
    # square = cv2.warpAffine(square,M,(400,300))
    # cv2.imshow('Shearing'+w,square)
    # close_windows()
    pass


# 5.1 Load Cifar10 training dataset,
# and then show  9 Images(Pop-up) and Labels respectively (4%)
def hw5_1(): 
    # (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    # cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # fig = plt.figure()

    # ax = plt.subplot(331)
    # plt.title(str(cifar_classes[y_train[0][0]]))
    # plt.imshow(x_train[0])
    # plt.axis('off')

    # ax = plt.subplot(332)
    # plt.title(str(cifar_classes[y_train[1][0]]))
    # plt.imshow(x_train[1])
    # plt.axis('off')

    # ax = plt.subplot(333)
    # plt.title(str(cifar_classes[y_train[2][0]]))
    # plt.imshow(x_train[2])
    # plt.axis('off')

    # ax = plt.subplot(334)
    # plt.title(str(cifar_classes[y_train[3][0]]))
    # plt.imshow(x_train[3]) 
    # plt.axis('off')

    # ax = plt.subplot(335)
    # plt.title(str(cifar_classes[y_train[4][0]]))
    # plt.imshow(x_train[4])
    # plt.axis('off')

    # ax = plt.subplot(336)
    # plt.title(str(cifar_classes[y_train[5][0]]))
    # plt.imshow(x_train[5])
    # plt.axis('off')

    # ax = plt.subplot(337)
    # plt.title(str(cifar_classes[y_train[6][0]]))
    # plt.imshow(x_train[6]) 
    # plt.axis('off')

    # ax = plt.subplot(338)
    # plt.title(str(cifar_classes[y_train[7][0]]))
    # plt.imshow(x_train[7]) 
    # plt.axis('off')

    # ax = plt.subplot(339)
    # plt.title(str(cifar_classes[y_train[8][0]]))
    # plt.imshow(x_train[8]) 
    # plt.axis('off')

    # plt.show()
    # plt.close()
    pass

# 5.2 	Print out training hyperparameters on the terminal
def hw5_2():
    print('hyperparameters:\nbatch size:128'+'\nlearning rate:0.1'+'\noptimizer:SGD')

def hw5_3():

    # model=tensorflow.keras.models.load_model('CIFAR10_model.h5')
    # print(model.summary()) # show output shape
    pass


# 5.4 take a screenshot of your training loss and accuracy
def hw5_4():
    # acc=cv2.imread('accuracy_50_epochs.png')
    # cv2.imshow('Training loss and accuracy.'+w,acc)
    # close_windows()
    pass

# let us choose one image from test images, inference the image, show the result image and class
def hw5_5(num=0):
    # model=tensorflow.keras.models.load_model('CIFAR10_model.h5')
    # (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    # cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # result=model.predict(x_train[num:num+1])
    
    # x = np.arange(len(cifar10_classes))
    # fig = plt.figure(figsize=(20,5))  

    # plt.subplot(121)
    # plt.title(str(cifar10_classes[y_train[num][0]]))
    # plt.imshow(x_train[num])   

    # plt.subplot(122)
    # plt.bar(x, result[0])
    # plt.xticks(x, cifar10_classes)
    # plt.xlabel('cifar10_classes')
    # plt.ylabel('%')
    # plt.show()
    # plt.close()
    pass


#if __name__ == "__main__":
#    hw5_5(0)