import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk

root = tk.Tk()
root.withdraw() 

#Z = baseline * f / (d + doffs)
baseline=342.789 #mm
focal_length=4019.284 #pixel
doffs=279.184 #pixel

class Question3:
    def __init__(self):
        self.imgL0 = None
        self.imgR0 = None
        self.imgL = None
        self.imgR = None
        self.disparity = None
        self.stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

    # 3-1: Stereo disparity map
    def stereo_disparity_map(self, imgL_path, imgR_path):
        self.imgL0 = cv2.imread(imgL_path, 0)
        self.imgR0 = cv2.imread(imgR_path, 0)

        self.disparity = self.stereo.compute(self.imgL0, self.imgR0)
        self.disparity = cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.imgL = cv2.imread(imgL_path)
        self.imgR = cv2.imread(imgR_path)

        cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
        cv2.moveWindow('imgL', 0, 200)
        cv2.imshow('imgL', self.imgL)
        self.resize_and_maintain_aspect_ratio('imgL', self.imgL.shape[1], self.imgL.shape[0], int(self.screen_width/3))

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.moveWindow('disparity', int(2*self.screen_width/3), 200)
        cv2.imshow('disparity', self.disparity)
        self.resize_and_maintain_aspect_ratio('disparity', self.disparity.shape[1], self.disparity.shape[0], int(self.screen_width/3))

        cv2.setMouseCallback('imgL', self.draw_circle)
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

            imgR_dot = self.imgR.copy()
            z=img[y][x][0]
            
            if img[y][x][0] != 0:       
                cv2.circle(imgR_dot,(x-z,y),25,(0,255,0),-1)

            cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
            cv2.moveWindow('imgR_dot', int(self.screen_width/3), 200)
            cv2.imshow('imgR_dot', imgR_dot)
            self.resize_and_maintain_aspect_ratio('imgR_dot', imgR_dot.shape[1], imgR_dot.shape[0], int(self.screen_width/3))
            cv2.waitKey(0)
    
    def resize_and_maintain_aspect_ratio(self, winName, oldWidth, oldHeight, newWidth):
        new_height = int(oldHeight * (newWidth / oldWidth))
        cv2.resizeWindow(winName, newWidth, new_height)