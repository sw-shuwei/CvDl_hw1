# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'opencvdl_hw1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QLineEdit
import hwFunc as func
from Q1 import Question1
from Q2 import Question2
from Q3 import Question3
from Q4 import Question4

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 30, 750, 750))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        
        verticalLayout = QVBoxLayout(self.gridLayoutWidget)
        verticalLayout.setContentsMargins(0, 0, 0, 0)
        verticalLayout.setObjectName("verticalLayout")
        self.upperHorizontalLayout = QHBoxLayout(self.gridLayoutWidget)
        self.upperHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.upperHorizontalLayout.setObjectName("upperHorizontalLayout")
        verticalLayout.addLayout(self.upperHorizontalLayout)
        self.lowerHorizontalLayout = QHBoxLayout(self.gridLayoutWidget)
        self.lowerHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.lowerHorizontalLayout.setObjectName("lowerHorizontalLayout")
        verticalLayout.addLayout(self.lowerHorizontalLayout)

        self.block_0 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_0.setObjectName("block_0")

        self.verticalLayout_0 = QtWidgets.QVBoxLayout(self.block_0)
        self.verticalLayout_0.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_0.setObjectName("verticalLayout_0")

        self.open_folder_btn = QPushButton('Load Folder', self.centralwidget)
        self.open_folder_btn.clicked.connect(self.showFolderDialog)
        self.verticalLayout_0.addWidget(self.open_folder_btn)

        self.open_image_L_btn = QPushButton('Load Image_L', self.centralwidget)
        self.open_image_L_btn.clicked.connect(self.showFileDialog)
        self.verticalLayout_0.addWidget(self.open_image_L_btn)

        self.open_image_R_btn = QPushButton('Load Image_R', self.centralwidget)
        self.open_image_R_btn.clicked.connect(self.showFileDialog)
        self.verticalLayout_0.addWidget(self.open_image_R_btn)
        self.upperHorizontalLayout.addWidget(self.block_0)

        self.block_1 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_1.setObjectName("block_1")

        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.block_1)
        self.verticalLayout_1.setContentsMargins(4, 4, 4, 4)
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

        self.block_2 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_2.setObjectName("block_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.block_2)
        self.verticalLayout_2.setContentsMargins(4, 4, 4, 4)
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

        self.block_3 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_3.setObjectName("block_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.block_3)
        self.verticalLayout_3.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stereo_display_map_btn = QtWidgets.QPushButton(self.block_3)
        self.stereo_display_map_btn.setObjectName("stereo_display_map_btn")
        self.verticalLayout_3.addWidget(self.stereo_display_map_btn)
        self.upperHorizontalLayout.addWidget(self.block_3)

        self.block_4 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_4.setObjectName("block_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.block_4)
        self.verticalLayout_4.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_4.setObjectName("verticalLayout_5")
        self.HW4_1 = QtWidgets.QPushButton(self.block_4)
        self.HW4_1.setObjectName("HW4_1")
        self.verticalLayout_4.addWidget(self.HW4_1)
        self.HW4_2 = QtWidgets.QPushButton(self.block_4)
        self.HW4_2.setObjectName("HW4_2")
        self.verticalLayout_4.addWidget(self.HW4_2)
        self.keypoints_btn = QtWidgets.QPushButton(self.block_4)
        self.keypoints_btn.setObjectName("keypoints_btn")
        self.verticalLayout_4.addWidget(self.keypoints_btn)
        self.matched_keypoints_btn = QtWidgets.QPushButton(self.block_4)
        self.matched_keypoints_btn.setObjectName("matched_keypoints_btn")
        self.verticalLayout_4.addWidget(self.matched_keypoints_btn)
        self.lowerHorizontalLayout.addWidget(self.block_4)

        self.block_5 = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.block_5.setObjectName("block_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.block_5)
        self.verticalLayout_5.setContentsMargins(-1, 8, -1, 8)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.HW5_1 = QtWidgets.QPushButton(self.block_5)
        self.HW5_1.setObjectName("HW5_1")
        self.verticalLayout_5.addWidget(self.HW5_1)
        self.HW5_2 = QtWidgets.QPushButton(self.block_5)
        self.HW5_2.setObjectName("HW5_2")
        self.verticalLayout_5.addWidget(self.HW5_2)
        self.HW5_3 = QtWidgets.QPushButton(self.block_5)
        self.HW5_3.setObjectName("HW5_3")
        self.verticalLayout_5.addWidget(self.HW5_3)
        self.HW5_4 = QtWidgets.QPushButton(self.block_5)
        self.HW5_4.setObjectName("HW5_4")
        self.verticalLayout_5.addWidget(self.HW5_4)
        self.HW5_5 = QtWidgets.QPushButton(self.block_5)
        self.HW5_5.setObjectName("HW5_5")
        self.verticalLayout_5.addWidget(self.HW5_5)
        self.lowerHorizontalLayout.addWidget(self.block_5)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # button clicked
        self.find_corners_btn.clicked.connect(Q1.find_corners) # 1-1
        self.find_intrinsic_matrix_btn.clicked.connect(Q1.find_intrinsic_matrix) # 1-2
        self.find_extrinsic_matrix_btn.clicked.connect(lambda: Q1.find_extrinsic_matrix(self.choose_bmp_img.currentText())) # 1-3
        self.find_distortion_matrix_btn.clicked.connect(Q1.find_distortion_matrix) # 1-4
        self.show_undistorted_result_btn.clicked.connect(Q1.show_undistorted_result) # 1-5
        self.show_words_on_boards_btn.clicked.connect(lambda: Q2.show_words_on_board(self.input_words.text())) # 2-1
        self.show_words_vertically_btn.clicked.connect(lambda: Q2.show_words_vertically(self.input_words.text())) # 2-2
        self.stereo_display_map_btn.clicked.connect(Q3.stereoDisparityMap)

        self.HW4_1.clicked.connect(func.hw4_1)
        self.HW4_2.clicked.connect(func.hw4_2)
        self.keypoints_btn.clicked.connect(Q4.keypoints)
        self.matched_keypoints_btn.clicked.connect(Q4.matched_keypoints)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def showFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_name, _ = file_dialog.getOpenFileName(self, 'Open File', '', 'Text Files (*.txt);;All Files (*)', options=options)

        if file_name:
            print('Selected file:', file_name)

    def showFolderDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly  # 只顯示目錄

        folder_dialog = QFileDialog()
        folder_name = folder_dialog.getExistingDirectory(MainWindow, 'Open Folder', '', options=options)

        if folder_name:
            print('Selected folder:', folder_name)


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
        self.HW4_1.setText(_translate("MainWindow", "Load Image1"))
        self.HW4_2.setText(_translate("MainWindow", "Load Image2"))
        self.keypoints_btn.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.matched_keypoints_btn.setText(_translate("MainWindow", "4.2 Matched Keypoints"))

        self.block_5.setTitle(_translate("MainWindow", "  5. VGG19"))
        self.HW5_1.setText(_translate("MainWindow", "Load Image"))
        self.HW5_2.setText(_translate("MainWindow", "5.1 Show Augmented Image"))
        self.HW5_3.setText(_translate("MainWindow", "5.2 Show Model Structures"))
        self.HW5_4.setText(_translate("MainWindow", "5.3 Show Acc and Loss"))
        self.HW5_5.setText(_translate("MainWindow", "5.4 Inference"))



if __name__ == "__main__":
    import sys
    Q1 = Question1()
    Q2 = Question2()
    Q3 = Question3()
    Q4 = Question4()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
