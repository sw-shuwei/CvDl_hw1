import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
#Z = baseline * f / (d + doffs)
baseline=342.789 #mm
focal_length=4019.284 #pixel
doffs=279.184 #pixel

class Question3:
    def __init__(self):
        imgL = cv2.imread('Q3_image/imL.png',0)
        imgR = cv2.imread('Q3_image/imR.png',0)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(imgL,imgR)
        self.disparity = cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    def stereoDisparityMap(self):
        imgL = cv2.imread('Q3_image/imL.png')
        
        cv2.namedWindow('imgL', cv2.WINDOW_AUTOSIZE)
        # cv2.resizeWindow("imgL", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
        cv2.setMouseCallback('imgL', self.draw_circle)
        cv2.imshow('imgL',imgL)

        cv2.namedWindow('disparity', cv2.WINDOW_AUTOSIZE)
        # cv2.resizeWindow("disparity", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
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

            imgR_dot = cv2.imread('Q3_image/imR.png')
            z=img[y][x][0]
            
            if img[y][x][0] != 0:       
                cv2.circle(imgR_dot,(x-z,y),25,(0,255,0),-1)
            
            cv2.namedWindow('imgR_dot', cv2.WINDOW_AUTOSIZE)
            # cv2.resizeWindow("imgR_dot", int(imgR_dot.shape[1]/4), int(imgR_dot.shape[0]/4))
            cv2.imshow('imgR_dot', imgR_dot)           
            cv2.waitKey(0)