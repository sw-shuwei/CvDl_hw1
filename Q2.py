import cv2
import numpy as np
import glob

class Question2:
    def __init__(self):
        self.word = None
        self.floder = None

    # HW2-1: show words on board
    def show_words_on_board(self, floder, text):
        fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        self.floder = floder
        self.word = text
        self.chessboard_AR(fs)

    # HW2-2: show words vertically on board
    def show_words_vertically(self, floder, text):
        fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        self.floder = floder
        self.word = text
        self.chessboard_AR(fs)

    def show_words(self, img, lines, rvecs, tvecs, mtx, dist):
        # Space for showing words
        space = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        # Shift the word to the Space
        shift = np.ndarray.tolist(np.zeros(len(self.word)))
        for i in range(len(lines)):
            shift[i] = np.float32(lines[i] + space[i]).reshape(-1, 3)
        shift = np.concatenate(tuple(shift), axis=0)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(shift, rvecs, tvecs, mtx, dist)

        imgpts = imgpts.reshape(int(imgpts.shape[0] / 2), 2, 2).astype(int)

        # draw lines
        for l in range(len(imgpts)):
            img = cv2.line(img, tuple(imgpts[l][0]), tuple(imgpts[l][1]), (0, 0, 255), 2)

    def chessboard_AR(self, fs):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(self.floder+'/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
                # Find the rotation and translation vectors.
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

                # get data from library
                alphabet = np.ndarray.tolist(np.zeros(len(self.word)))
                for w in range(len(self.word)):
                    alphabet[w] = fs.getNode(self.word[w]).mat()

                # Show word on chessboard
                self.show_words(img, alphabet, rvecs, tvecs, mtx, dist)

                cv2.imshow('AR', img)
                cv2.waitKey(1000)
            cv2.destroyWindow('AR')
