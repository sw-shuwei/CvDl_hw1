# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'opencvdl_hw1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QLineEdit, QLabel
from Q1 import Question1
from Q2 import Question2
from Q3 import Question3
from Q4 import Question4
from Q5 import Question5

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 535)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.centralwidget.setGeometry(QtCore.QRect(20, 30, 750, 750))

        self.selected_img = 'Q5_image/Q5_4/frog.png'
        self.selected_imgL = 'Q3_image/imL.png'
        self.selected_imgR = 'Q3_image/imR.png'
        self.selected_img1 = 'Q4_Image/Left.jpg'
        self.selected_img2 = 'Q4_Image/Right.jpg'
        self.selected_floder = 'Q2_Image'
        
        verticalLayout = QVBoxLayout(self.centralwidget)
        verticalLayout.setContentsMargins(-1, 8, -1, 8)
        verticalLayout.setObjectName("verticalLayout")
        self.upperHorizontalLayout = QHBoxLayout(self.centralwidget)
        self.upperHorizontalLayout.setContentsMargins(-1, 8, -1, 8)
        self.upperHorizontalLayout.setObjectName("upperHorizontalLayout")
        verticalLayout.addLayout(self.upperHorizontalLayout)
        self.lowerHorizontalLayout = QHBoxLayout(self.centralwidget)
        self.lowerHorizontalLayout.setContentsMargins(-1, 8, -1, 8)
        self.lowerHorizontalLayout.setObjectName("lowerHorizontalLayout")
        verticalLayout.addLayout(self.lowerHorizontalLayout)

        self.block_0 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_0.setObjectName("block_0")

        self.verticalLayout_0 = QtWidgets.QVBoxLayout(self.block_0)
        self.verticalLayout_0.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_0.setObjectName("verticalLayout_0")

        self.open_folder_btn = QPushButton('Load Folder', self.centralwidget)
        self.open_folder_btn.clicked.connect(self.showFolderDialog)
        self.verticalLayout_0.addWidget(self.open_folder_btn)

        self.open_image_L_btn = QPushButton('Load Image_L', self.centralwidget)
        self.open_image_L_btn.clicked.connect(self.load_imgL)
        self.verticalLayout_0.addWidget(self.open_image_L_btn)

        self.open_image_R_btn = QPushButton('Load Image_R', self.centralwidget)
        self.open_image_R_btn.clicked.connect(self.load_imgR)
        self.verticalLayout_0.addWidget(self.open_image_R_btn)
        self.upperHorizontalLayout.addWidget(self.block_0)

        self.block_1 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_1.setObjectName("block_1")

        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.block_1)
        self.verticalLayout_1.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_1.setObjectName("verticalLayout_1")

        self.find_corners_btn = QtWidgets.QPushButton(self.block_1)
        self.find_corners_btn.setObjectName("find_corners_btn")
        self.verticalLayout_1.addWidget(self.find_corners_btn)

        self.find_intrinsic_matrix_btn = QtWidgets.QPushButton(self.block_1)
        self.find_intrinsic_matrix_btn.setObjectName("find_intrinsic_matrix_btn")
        self.verticalLayout_1.addWidget(self.find_intrinsic_matrix_btn)
        
        self.group_1_3 = QtWidgets.QGroupBox(self.block_1)
        self.group_1_3.setObjectName("group_1_3")

        self.find_extrinsic_matrix_btn = QtWidgets.QPushButton(self.group_1_3)
        self.find_extrinsic_matrix_btn.setObjectName("find_extrinsic_matrix_btn")

        self.choose_bmp_img = QComboBox(self.group_1_3)
        self.choose_bmp_img.setObjectName("choose_bmp_img")

        for i in range(1, 16):
            self.choose_bmp_img.addItem(str(i))

        self.verticalLayout_1_3 = QtWidgets.QVBoxLayout(self.group_1_3)
        self.verticalLayout_1_3.setObjectName("verticalLayout_1_3")

        self.verticalLayout_1_3.addWidget(self.choose_bmp_img)
        self.verticalLayout_1_3.addWidget(self.find_extrinsic_matrix_btn)
        self.verticalLayout_1.addWidget(self.group_1_3)

        self.find_distortion_matrix_btn = QtWidgets.QPushButton(self.block_1)
        self.find_distortion_matrix_btn.setObjectName("find_distortion_matrix_btn")
        self.verticalLayout_1.addWidget(self.find_distortion_matrix_btn)

        self.show_undistorted_result_btn = QtWidgets.QPushButton(self.block_1)
        self.show_undistorted_result_btn.setObjectName("show_undistorted_result_btn")
        self.verticalLayout_1.addWidget(self.show_undistorted_result_btn)
        self.upperHorizontalLayout.addWidget(self.block_1)

        self.block_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_2.setObjectName("block_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.block_2)
        self.verticalLayout_2.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.input_words = QLineEdit(self.block_2)
        self.input_words.setObjectName("input_words")
        self.verticalLayout_2.addWidget(self.input_words)

        self.show_words_on_boards_btn = QtWidgets.QPushButton(self.block_2)
        self.show_words_on_boards_btn.setObjectName("show_words_on_boards_btn")
        self.verticalLayout_2.addWidget(self.show_words_on_boards_btn)

        self.show_words_vertically_btn = QtWidgets.QPushButton(self.block_2)
        self.show_words_vertically_btn.setObjectName("show_words_vertically_btn")
        self.verticalLayout_2.addWidget(self.show_words_vertically_btn)
        self.upperHorizontalLayout.addWidget(self.block_2)

        self.block_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_3.setObjectName("block_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.block_3)
        self.verticalLayout_3.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stereo_display_map_btn = QtWidgets.QPushButton(self.block_3)
        self.stereo_display_map_btn.setObjectName("stereo_display_map_btn")
        self.verticalLayout_3.addWidget(self.stereo_display_map_btn)
        self.upperHorizontalLayout.addWidget(self.block_3)

        self.block_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_4.setObjectName("block_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.block_4)
        self.verticalLayout_4.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.load_img1_btn = QtWidgets.QPushButton(self.block_4)
        self.load_img1_btn.setObjectName("load_img1_btn")
        self.verticalLayout_4.addWidget(self.load_img1_btn)
        self.load_img2_btn = QtWidgets.QPushButton(self.block_4)
        self.load_img2_btn.setObjectName("load_img2_btn")
        self.verticalLayout_4.addWidget(self.load_img2_btn)
        self.keypoints_btn = QtWidgets.QPushButton(self.block_4)
        self.keypoints_btn.setObjectName("keypoints_btn")
        self.verticalLayout_4.addWidget(self.keypoints_btn)
        self.matched_keypoints_btn = QtWidgets.QPushButton(self.block_4)
        self.matched_keypoints_btn.setObjectName("matched_keypoints_btn")
        self.verticalLayout_4.addWidget(self.matched_keypoints_btn)
        self.lowerHorizontalLayout.addWidget(self.block_4)

        self.block_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.block_5.setObjectName("block_5")
        self.img_and_btns_5 = QtWidgets.QHBoxLayout(self.block_5)
        self.img_and_btns_5.setObjectName("img_and_btns_5")
        self.verticalLayout_5_btn = QtWidgets.QVBoxLayout(self.block_5)
        self.verticalLayout_5_btn.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_5_btn.setObjectName("verticalLayout_5_btn")
        self.verticalLayout_5_pred_img = QtWidgets.QVBoxLayout(self.block_5)
        self.verticalLayout_5_pred_img.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_5_pred_img.setObjectName("verticalLayout_5_pred_img")

        self.load_image_btn = QtWidgets.QPushButton(self.block_5)
        self.load_image_btn.setObjectName("load_image_btn")
        self.show_augumentation_imgs_btn = QtWidgets.QPushButton(self.block_5)
        self.show_augumentation_imgs_btn.setObjectName("show_augumentation_imgs_btn")
        self.show_model_structure_btn = QtWidgets.QPushButton(self.block_5)
        self.show_model_structure_btn.setObjectName("show_model_structure_btn")
        self.show_acc_loss_btn = QtWidgets.QPushButton(self.block_5)
        self.show_acc_loss_btn.setObjectName("show_acc_loss_btn")
        self.inference_btn = QtWidgets.QPushButton(self.block_5)
        self.inference_btn.setObjectName("inference_btn")
        self.test_img = QtWidgets.QLabel(self.block_5)
        self.test_img.setObjectName("test_img")
        self.pred_result = QLabel()
        self.pred_result.setText("")
        
        self.btn = QtWidgets.QWidget()
        self.img = QtWidgets.QWidget()
        btn_layout = QtWidgets.QVBoxLayout(self.btn)
        img_layout = QtWidgets.QVBoxLayout(self.img)

        # Add buttons to btn_layout
        btn_layout.addWidget(self.load_image_btn)
        btn_layout.addWidget(self.show_augumentation_imgs_btn)
        btn_layout.addWidget(self.show_model_structure_btn)
        btn_layout.addWidget(self.show_acc_loss_btn)
        btn_layout.addWidget(self.inference_btn)

        img_layout.addWidget(self.test_img)
        img_layout.addWidget(self.pred_result)

        self.img_and_btns_5.addWidget(self.btn)
        self.img_and_btns_5.addWidget(self.img)
        self.lowerHorizontalLayout.addWidget(self.block_5)

        self.upperHorizontalLayout.setStretch(0, 1)  # block_0
        self.upperHorizontalLayout.setStretch(1, 1)  # block_1
        self.upperHorizontalLayout.setStretch(2, 1)  # block_2
        self.upperHorizontalLayout.setStretch(3, 1)  # block_3
        self.lowerHorizontalLayout.setStretch(0, 1)  # block_4
        self.lowerHorizontalLayout.setStretch(1, 1)  # block_5

        MainWindow.setCentralWidget(self.centralwidget)

        # button clicked
        self.find_corners_btn.clicked.connect(Q1.find_corners)                       # 1-1
        self.find_intrinsic_matrix_btn.clicked.connect(Q1.find_intrinsic_matrix)     # 1-2
        self.find_extrinsic_matrix_btn.clicked.connect(
            lambda: Q1.find_extrinsic_matrix(self.choose_bmp_img.currentText()))     # 1-3
        self.find_distortion_matrix_btn.clicked.connect(Q1.find_distortion_matrix)   # 1-4
        self.show_undistorted_result_btn.clicked.connect(Q1.show_undistorted_result) # 1-5
        self.show_words_on_boards_btn.clicked.connect(
            lambda: Q2.show_words_on_board(self.selected_floder, self.input_words.text()))   # 2-1
        self.show_words_vertically_btn.clicked.connect(
            lambda: Q2.show_words_vertically(self.selected_floder, self.input_words.text())) # 2-2
        self.stereo_display_map_btn.clicked.connect(self.stereo_display_map) # 3-1

        self.load_img1_btn.clicked.connect(self.load_img1)                 # 4-0
        self.load_img2_btn.clicked.connect(self.load_img2)                 # 4-0
        self.keypoints_btn.clicked.connect(self.keypoints)                 # 4-1
        self.matched_keypoints_btn.clicked.connect(self.matched_keypoints) # 4-2

        self.load_image_btn.clicked.connect(self.load_img_to_inference)              # 5-0
        self.show_augumentation_imgs_btn.clicked.connect(Q5.show_augumentation_imgs) # 5-1
        self.show_model_structure_btn.clicked.connect(Q5.show_model_structure)       # 5-2
        self.show_acc_loss_btn.clicked.connect(Q5.show_acc_loss)                     # 5-3
        self.inference_btn.clicked.connect(self.inference)                           # 5-4
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def load_imgL(self):
        self.selected_imgL = self.showFileDialog()

    def load_imgR(self):
        self.selected_imgR = self.showFileDialog()
    
    def load_img1(self):
        self.selected_img1 = self.showFileDialog()
    
    def load_img2(self):
        self.selected_img2 = self.showFileDialog()

    def load_img_to_inference(self):
        self.selected_img = self.showFileDialog()

    def showFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        selected_img, _ = file_dialog.getOpenFileName(MainWindow, 'Open File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)', options=options)

        if selected_img: 
            print('Selected file:', selected_img)
            return selected_img
        return None

    def showFolderDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly  # 只顯示目錄

        folder_dialog = QFileDialog()
        selected_folder = folder_dialog.getExistingDirectory(MainWindow, 'Open Folder', '', options=options)

        if selected_folder:
            self.selected_floder = selected_folder
            print('Selected folder:', selected_folder)
    
    def stereo_display_map(self):
        if self.selected_imgL and self.selected_imgR:
            Q3.stereo_disparity_map(self.selected_imgL, self.selected_imgR)

    def keypoints(self):
        if self.selected_img1:
            Q4.keypoints(self.selected_img1)

    def matched_keypoints(self):
        if self.selected_img1 and self.selected_img2:
            Q4.matched_keypoints(self.selected_img1, self.selected_img2)

    def inference(self):
        if self.selected_img:
            pixmap = QPixmap(self.selected_img)
            self.resize_and_display_image(pixmap)
            pred = Q5.inference(self.selected_img)
            self.pred_result.setText('Predict: '+ pred)

    def resize_and_display_image(self, pixmap):
        pixmap = pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio)
        self.test_img.setPixmap(pixmap)
        self.test_img.setScaledContents(True)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow-cvdlhw1"))

        self.block_0.setTitle(_translate("MainWindow", "  Load Image"))
        
        self.block_1.setTitle(_translate("MainWindow", "  1. Calibration"))
        self.find_corners_btn.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.find_intrinsic_matrix_btn.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.group_1_3.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.find_extrinsic_matrix_btn.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.find_distortion_matrix_btn.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.show_undistorted_result_btn.setText(_translate("MainWindow", "1.5 Show Result"))


        self.block_2.setTitle(_translate("MainWindow", "  2. Augmented Reality"))
        self.show_words_on_boards_btn.setText(_translate("MainWindow", "2.1 Show Words on Boards"))
        self.show_words_vertically_btn.setText(_translate("MainWindow", "2.2 Show Words Vertically"))

        self.block_3.setTitle(_translate("MainWindow", "  3. Stereo Disparity Map"))
        self.stereo_display_map_btn.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))

        self.block_4.setTitle(_translate("MainWindow", "  4. SIFT"))
        self.load_img1_btn.setText(_translate("MainWindow", "Load Image 1"))
        self.load_img2_btn.setText(_translate("MainWindow", "Load Image 2"))
        self.keypoints_btn.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.matched_keypoints_btn.setText(_translate("MainWindow", "4.2 Matched Keypoints"))

        self.block_5.setTitle(_translate("MainWindow", "  5. VGG19"))
        self.load_image_btn.setText(_translate("MainWindow", "Load Image"))
        self.show_augumentation_imgs_btn.setText(_translate("MainWindow", "5.1 Show Augmented Image"))
        self.show_model_structure_btn.setText(_translate("MainWindow", "5.2 Show Model Structures"))
        self.show_acc_loss_btn.setText(_translate("MainWindow", "5.3 Show Acc and Loss"))
        self.inference_btn.setText(_translate("MainWindow", "5.4 Inference"))


if __name__ == "__main__":
    import sys
    Q1 = Question1()
    Q2 = Question2()
    Q3 = Question3()
    Q4 = Question4()
    Q5 = Question5()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
