from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap

import SimpleITK as sitk

import numpy as np
import cv2

from skimage import filters
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import skeletonize_3d

import qimage2ndarray

import scipy
import scipy.ndimage
import scipy.fftpack as fftim
import scipy.misc

import math

import Canny1
import Renyi1
import watershed1
import hit_or_miss1

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 430)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 두개의 graphicsView를 사용
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(50, 80, 260, 260))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(400, 80, 260, 260))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 40, 181, 31))
        #폰트와 크기 조정
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(500, 40, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        #up, down, reset 버튼 사용
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(700, 190, 61, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 240, 61, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(700, 140, 61, 28))
        self.pushButton_10.setObjectName("pushButton_10")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuFile_2 = QtWidgets.QMenu(self.menubar)
        self.menuFile_2.setObjectName("menuFile_2")
        self.menuFile_3 = QtWidgets.QMenu(self.menubar)
        self.menuFile_3.setObjectName("menuFile_3")
        self.menuFile_4 = QtWidgets.QMenu(self.menubar)
        self.menuFile_4.setObjectName("menuFile_4")
        self.menuFile_5 = QtWidgets.QMenu(self.menubar)
        self.menuFile_5.setObjectName("menuFile_5")
        self.Lowpass_Filter = QtWidgets.QMenu(self.menuFile_4)
        self.Lowpass_Filter.setObjectName("Lowpass_Filter")
        self.Highpass_Filter = QtWidgets.QMenu(self.menuFile_4)
        self.Highpass_Filter.setObjectName("Highpass_Filter")
        self.menuFile_6 = QtWidgets.QMenu(self.menubar)
        self.menuFile_6.setObjectName("menuFile_6")
        MainWindow.setMenuBar(self.menubar)

        self.actionOPEN = QtWidgets.QAction(MainWindow)
        self.actionOPEN.setObjectName("actionOPEN")
        self.actionCONVERT = QtWidgets.QAction(MainWindow)
        self.actionCONVERT.setObjectName("actionCONVERT")
        self.mean = QtWidgets.QAction(MainWindow)
        self.mean.setObjectName("mean")
        self.max = QtWidgets.QAction(MainWindow)
        self.max.setObjectName("max")
        self.median = QtWidgets.QAction(MainWindow)
        self.median.setObjectName("median")
        self.sobel = QtWidgets.QAction(MainWindow)
        self.sobel.setObjectName("sobel")
        self.prewitt = QtWidgets.QAction(MainWindow)
        self.prewitt.setObjectName("prewitt")
        self.canny = QtWidgets.QAction(MainWindow)
        self.canny.setObjectName("canny")
        self.log = QtWidgets.QAction(MainWindow)
        self.log.setObjectName("log")
        self.image_inverse = QtWidgets.QAction(MainWindow)
        self.image_inverse.setObjectName("image_inverse")
        self.power_law_transformation = QtWidgets.QAction(MainWindow)
        self.power_law_transformation.setObjectName("power_law_transformation")
        self.log_transformation = QtWidgets.QAction(MainWindow)
        self.log_transformation.setObjectName("log_transformation")
        self.histogram_equalization = QtWidgets.QAction(MainWindow)
        self.histogram_equalization.setObjectName("histogram_equalization")
        self.contrast_stretching = QtWidgets.QAction(MainWindow)
        self.contrast_stretching.setObjectName("contrast_stretching")
        self.Bandpass_Filter = QtWidgets.QAction(MainWindow)
        self.Bandpass_Filter.setObjectName("Bandpass_Filter")
        self.otsu_method = QtWidgets.QAction(MainWindow)
        self.otsu_method.setObjectName("otsu_method")
        self.renyi_entropy = QtWidgets.QAction(MainWindow)
        self.renyi_entropy.setObjectName("renyi_entropy")
        self.adaptive_thresholding = QtWidgets.QAction(MainWindow)
        self.adaptive_thresholding.setObjectName("adaptive_thresholding")
        self.watershed_segmentation = QtWidgets.QAction(MainWindow)
        self.watershed_segmentation.setObjectName("watershed_segmentation")
        self.dilation = QtWidgets.QAction(MainWindow)
        self.dilation.setObjectName("dilation")
        self.erosion = QtWidgets.QAction(MainWindow)
        self.erosion.setObjectName("erosion")
        self.opening = QtWidgets.QAction(MainWindow)
        self.opening.setObjectName("opening")
        self.closing = QtWidgets.QAction(MainWindow)
        self.closing.setObjectName("closing")
        self.hit_or_miss = QtWidgets.QAction(MainWindow)
        self.hit_or_miss.setObjectName("hit_or_miss")
        self.skeletonize = QtWidgets.QAction(MainWindow)
        self.skeletonize.setObjectName("skeletonize")


        #상위 메뉴바와 하위 메뉴바를 연결
        self.Ideal_Lowpass_Filter = QtWidgets.QAction(MainWindow)
        self.Ideal_Lowpass_Filter.setObjectName("Ideal_Lowpass_Filter")
        self.Ideal_Highpass_Filter = QtWidgets.QAction(MainWindow)
        self.Ideal_Highpass_Filter.setObjectName("Ideal_Highpass_Filter")
        self.Butterworth_Lowpass_Filter = QtWidgets.QAction(MainWindow)
        self.Butterworth_Lowpass_Filter.setObjectName("Butterworth_Lowpass_Filter")
        self.Butterworth_Highpass_Filter = QtWidgets.QAction(MainWindow)
        self.Butterworth_Highpass_Filter.setObjectName("Butterworth_Highpass_Filter")
        self.Gaussian_Lowpass_Filter = QtWidgets.QAction(MainWindow)
        self.Gaussian_Lowpass_Filter.setObjectName("Gaussian_Lowpass_Filter")
        self.Gaussian_Highpass_Filter = QtWidgets.QAction(MainWindow)
        self.Gaussian_Highpass_Filter.setObjectName("Gaussian_Highpass_Filter")

        self.Lowpass_Filter.addAction(self.Ideal_Lowpass_Filter)
        self.Highpass_Filter.addAction(self.Ideal_Highpass_Filter)
        self.Lowpass_Filter.addAction(self.Butterworth_Lowpass_Filter)
        self.Highpass_Filter.addAction(self.Butterworth_Highpass_Filter)
        self.Lowpass_Filter.addAction(self.Gaussian_Lowpass_Filter)
        self.Highpass_Filter.addAction(self.Gaussian_Highpass_Filter)

        self.menuFile.addAction(self.actionOPEN)
        self.menuFile.addAction(self.actionCONVERT)
        self.menuFile_2.addAction(self.mean)
        self.menuFile_2.addAction(self.max)
        self.menuFile_2.addAction(self.median)
        self.menuFile_2.addAction(self.sobel)
        self.menuFile_2.addAction(self.prewitt)
        self.menuFile_2.addAction(self.canny)
        self.menuFile_2.addAction(self.log)
        self.menuFile_3.addAction(self.image_inverse)
        self.menuFile_3.addAction(self.power_law_transformation)
        self.menuFile_3.addAction(self.log_transformation)
        self.menuFile_3.addAction(self.histogram_equalization)
        self.menuFile_3.addAction(self.contrast_stretching)
        self.menuFile_4.addAction(self.Lowpass_Filter.menuAction())
        self.menuFile_4.addAction(self.Highpass_Filter.menuAction())
        self.menuFile_4.addAction(self.Bandpass_Filter)
        self.menuFile_5.addAction(self.otsu_method)
        self.menuFile_5.addAction(self.renyi_entropy)
        self.menuFile_5.addAction(self.adaptive_thresholding)
        self.menuFile_5.addAction(self.watershed_segmentation)
        self.menuFile_6.addAction(self.dilation)
        self.menuFile_6.addAction(self.erosion)
        self.menuFile_6.addAction(self.opening)
        self.menuFile_6.addAction(self.closing)
        self.menuFile_6.addAction(self.hit_or_miss)
        self.menuFile_6.addAction(self.skeletonize)


        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuFile_2.menuAction())
        self.menubar.addAction(self.menuFile_3.menuAction())
        self.menubar.addAction(self.menuFile_4.menuAction())
        self.menubar.addAction(self.menuFile_5.menuAction())
        self.menubar.addAction(self.menuFile_6.menuAction())

        #signal과 slot을 연결
        self.actionOPEN.triggered.connect(self.open_trigger)
        self.actionCONVERT.triggered.connect(self.save_trigger)
        self.mean.triggered.connect(self.mean_click)
        self.max.triggered.connect(self.max_click)
        self.median.triggered.connect(self.median_click)
        self.sobel.triggered.connect(self.sobel_click)
        self.prewitt.triggered.connect(self.prewitt_click)
        self.canny.triggered.connect(self.canny_click)
        self.log.triggered.connect(self.log_click)
        self.image_inverse.triggered.connect(self.image_inverse_click)
        self.power_law_transformation.triggered.connect(self.power_law_transformation_click)
        self.log_transformation.triggered.connect(self.log_transformation_click)
        self.histogram_equalization.triggered.connect(self.histogram_equalization_click)
        self.contrast_stretching.triggered.connect(self.contrast_stretching_click)
        self.pushButton.clicked.connect(self.up_click)
        self.pushButton_2.clicked.connect(self.down_click)
        self.pushButton_10.clicked.connect(self.reset_click)
        self.Ideal_Lowpass_Filter.triggered.connect(self.id_low_click)
        self.Ideal_Highpass_Filter.triggered.connect(self.id_high_click)
        self.Butterworth_Lowpass_Filter.triggered.connect(self.bu_low_click)
        self.Butterworth_Highpass_Filter.triggered.connect(self.bu_high_click)
        self.Gaussian_Lowpass_Filter.triggered.connect(self.ga_low_click)
        self.Gaussian_Highpass_Filter.triggered.connect(self.ga_high_click)
        self.Bandpass_Filter.triggered.connect(self.bandpass_click)
        self.otsu_method.triggered.connect(self.otsu_click)
        self.renyi_entropy.triggered.connect(self.renyi_click)
        self.adaptive_thresholding.triggered.connect(self.adaptive_click)
        self.watershed_segmentation.triggered.connect(self.watershed_click)
        self.dilation.triggered.connect(self.dilation_click)
        self.erosion.triggered.connect(self.erosion_click)
        self.opening.triggered.connect(self.opening_click)
        self.closing.triggered.connect(self.closing_click)
        self.hit_or_miss.triggered.connect(self.hit_or_miss_click)
        self.skeletonize.triggered.connect(self.skeletonize_click)



        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        #각자의 이름 설정
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MR DICOM GUI_soobum"))
        self.label.setText(_translate("MainWindow", "Before"))
        self.label_2.setText(_translate("MainWindow", "After"))
        self.pushButton.setText(_translate("MainWindow", "Up"))
        self.pushButton_2.setText(_translate("MainWindow", "Down"))
        self.pushButton_10.setText(_translate("MainWindow", "Reset"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuFile_2.setTitle(_translate("MainWindow", "Filter"))
        self.menuFile_3.setTitle(_translate("MainWindow", "Image Enhancement"))
        self.menuFile_4.setTitle(_translate("MainWindow", "Fourier Transform"))
        self.menuFile_5.setTitle(_translate("MainWindow", "Segmentation"))
        self.menuFile_6.setTitle(_translate("MainWindow", "Morphological"))
        self.actionOPEN.setText(_translate("MainWindow", "OPEN"))
        self.actionCONVERT.setText(_translate("MainWindow", "SAVE"))
        self.mean.setText(_translate("MainWindow", "Mean Filter"))
        self.max.setText(_translate("MainWindow", "Max Filter"))
        self.median.setText(_translate("MainWindow", "Median Filter"))
        self.sobel.setText(_translate("MainWindow", "Sobel Filter"))
        self.prewitt.setText(_translate("MainWindow", "Prewitt Filter"))
        self.canny.setText(_translate("MainWindow", "Canny Filter"))
        self.log.setText(_translate("MainWindow", "Log Filter"))
        self.image_inverse.setText(_translate("MainWindow", "Image Inverse"))
        self.power_law_transformation.setText(_translate("MainWindow", "Power Law Transformation"))
        self.log_transformation.setText(_translate("MainWindow", "Log Transformation"))
        self.histogram_equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.contrast_stretching.setText(_translate("MainWindow", "Contrast Stretching"))
        self.Lowpass_Filter.setTitle(_translate("MainWindow", "Lowpass Filter"))
        self.Highpass_Filter.setTitle(_translate("MainWindow", "Highpass Filter"))
        self.Ideal_Lowpass_Filter.setText(_translate("MainWindow", "Ideal Lowpass Filter"))
        self.Ideal_Highpass_Filter.setText(_translate("MainWindow", "Ideal Highpass Filter"))
        self.Butterworth_Lowpass_Filter.setText(_translate("MainWindow", "Butterworth Lowpass Filter"))
        self.Butterworth_Highpass_Filter.setText(_translate("MainWindow", "Butterworth Highpass Filter"))
        self.Gaussian_Lowpass_Filter.setText(_translate("MainWindow", "Gaussian Lowpass Filter"))
        self.Gaussian_Highpass_Filter.setText(_translate("MainWindow", "Gaussian Highpass Filter"))
        self.Bandpass_Filter.setText(_translate("MainWindow", "Bandpass Filter"))
        self.otsu_method.setText(_translate("MainWindow", "Otsu Method"))
        self.renyi_entropy.setText(_translate("MainWindow", "Renyi Entropy"))
        self.adaptive_thresholding.setText(_translate("MainWindow", "Adaptive Thresholding"))
        self.watershed_segmentation.setText(_translate("MainWindow", "Watershed Segmentation"))
        self.dilation.setText(_translate("MainWindow", "Dilation"))
        self.erosion.setText(_translate("MainWindow", "Erosion"))
        self.opening.setText(_translate("MainWindow", "Opening"))
        self.closing.setText(_translate("MainWindow", "Closing"))
        self.hit_or_miss.setText(_translate("MainWindow", "Hit_or_Miss"))
        self.skeletonize.setText(_translate("MainWindow", "Skeletonize"))


    def open_trigger(self):
        #Directory를 받아서 폴더자체를 가져오게 한다. 3D array
        fileDic = QFileDialog.getExistingDirectory(MainWindow, "Open a folder", "./", QFileDialog.ShowDirsOnly)
        #sitk = SimpleITK
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(fileDic)
        reader.SetFileNames(dicom_names)
        #reader.SetFileNames('C:\\Users\\ADmin\\Desktop\\의료영상 정리\\인턴\\dicom_viewer_Mrbrain/')
        self.image_open = reader.Execute()

        self.darray = sitk.GetArrayFromImage(self.image_open)
        #밑에 나오는 사이즈를 공통적으로 사용하기 위해  x, y, z를 self를 붙혀서 사용한다.
        (self.z, self.y, self.x) = self.darray.shape

        # nomalization
        for i in range(self.darray.shape[0]):
            self.darray[i] = (self.darray[i]-np.min(self.darray[i]))/(np.max(self.darray[i])-np.min(self.darray[i]))*255
        print(self.darray)
        darray_max = np.amax(self.darray)

        pixel_value = (self.darray / darray_max) * 255

        self.i = 0
        # convertScaleAbs는 float를 int형으로 변환해주는 함수
        self.cvuint8 = cv2.convertScaleAbs(pixel_value)
        self.image_slice = self.cvuint8[self.i, :, :]
        #graphicsView는 int값만 받기 때문에 int로 변환시켜준 것이고 이를 뿌려준다.
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_slice)))
        self.graphicsView.setScene(scene)


    def save_trigger(self):

        filepath = QFileDialog.getSaveFileName(MainWindow, 'Save File', '', "*.nii;;*.hdr")

        print(filepath)

        filename = filepath[0]

        print(filename)
        # save를 하면 180도 뒤집어져서 저장이 되기 때문에 np.rot90을 2번진행하여 180도를 회전시켜준다.
        rotated = np.rot90(self.pixel_value, 2)

        image = sitk.GetImageFromArray(rotated)

        sitk.WriteImage(image, filename)

    def up_click(self):
        #첫번째 graphicsview와 두번째가 동시에 up, down되도록 설정
        self.i = self.i + 1

        self.image_slice = self.cvuint8[self.i, :, :]
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_slice)))
        self.graphicsView.setScene(scene)

        self.image_after = self.after_cvuint8[self.i, :, :]

        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_after)))
        self.graphicsView_2.setScene(scene)


        #순환이 되도록 if문으로 구현
        if (self.i == self.z - 1 ):
            self.i = 0
        else:
            pass


    def down_click(self):

        self.i = self.i - 1

        self.image_slice = self.cvuint8[self.i, :, :]
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_slice)))
        self.graphicsView.setScene(scene)

        self.image_after = self.after_cvuint8[self.i, :, :]

        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_after)))
        self.graphicsView_2.setScene(scene)


        if (self.i == -(self.z + 1)):
            self.i = self.z - 1
        else:
            pass

    def reset_click(self):

        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_slice)))
        self.graphicsView_2.setScene(scene)


    def mean_click(self):

        sizevalue, r = QInputDialog.getInt(MainWindow, "Get Size", "Value:", 5, 0, 100)

        mean_mask = np.ones((sizevalue, sizevalue, sizevalue)) / sizevalue**3
        # 3D pixel > mean filtering
        self.pixel_value = scipy.ndimage.filters.convolve(self.darray, mean_mask)

        self.view()

    def max_click(self):

        sizevalue, r  = QInputDialog.getInt(MainWindow, "Get Size", "Value:", 3, 0, 100)
        # 3D pixel > max filtering
        self.pixel_value = scipy.ndimage.filters.maximum_filter(self.darray, size=sizevalue, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)

        self.view()


    def median_click(self):

        sizevalue, r = QInputDialog.getInt(MainWindow, "Get Size", "Value:", 3, 0, 100)
        # 3D pixel > median filtering
        self.pixel_value = scipy.ndimage.filters.median_filter(self.darray, size=sizevalue, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)

        self.view()

    def sobel_click(self):
        # 3D array sobel filtering

        self.pixel_value = scipy.ndimage.filters.sobel(self.darray)

        self.view()

    def prewitt_click(self):
        # 3D array prewitt filtering

        self.pixel_value = scipy.ndimage.filters.prewitt(self.darray)

        self.view()

    def canny_click(self):

        sigmavalue, r = QInputDialog.getDouble(MainWindow, "Get Sigma", "Value:", 0, 0, 100)
        # 21, 576, 576 > 576, 576, 21
        new = np.transpose(self.darray)
        gaussian = Canny1.gs_filter(new, sigmavalue)

        grad = Canny1.gradient_intensity(gaussian)
        # Too Large value
        darray_max = np.max(grad)
        print(darray_max)
        image = (grad[0] / darray_max) * 255
        direction = (grad[1] / darray_max) * 255

        supp1 = cv2.convertScaleAbs(image)
        supp2 = cv2.convertScaleAbs(direction)

        supp = Canny1.suppression(image, direction)

        thre = Canny1.threshold(supp, 50, 100)

        last = Canny1.tracking(thre[0], 50, strong=255)
        self.pixel_value = np.transpose(last)
        print(self.pixel_value)
        self.view()


    def log_click(self):

        sigmavalue, r = QInputDialog.getDouble(MainWindow, "Get Sigma", "Value:", 0, 0, 100)

        #3D array LoG filtering
        self.pixel_value = scipy.ndimage.filters.gaussian_laplace(self.darray, sigma = sigmavalue, mode='reflect')

        self.view()



    def image_inverse_click(self):
        self.darray = (self.darray/self.darray.max()) * 255
        self.pixel_value = 255 - self.darray
        self.view()

    def power_law_transformation_click(self):

        gammavalue, r = QInputDialog.getDouble(MainWindow, "Get Gamma", "Value:", 0, 0, 100)

        self.darray = self.darray + 1

        array_max = np.max(self.darray)
        nomalize = self.darray / array_max
        e = np.log(nomalize) * gammavalue
        self.pixel_value = np.exp(e) * 255

        self.view()



    def log_transformation_click(self):
        array_max = np.max(self.darray)

        print("--------------")
        #print(self.darray)
        print(np.min(self.darray))
        print(np.max(self.darray))
        print("--------------")
        # performing the log transformation
        self.pixel_value = (255.0 * np.log(1 + self.darray)) / (np.log(1 + array_max) + 1e-8)

        self.view()

    def histogram_equalization_click(self):
        #float에서 uint8로 변환, nomalize
        darray_max = np.max(self.darray)
        nomalize = self.darray * 255 / darray_max
        darray = cv2.convertScaleAbs(nomalize)

        fl = darray.flatten()

        hist, bins = np.histogram(darray, 256, [0, 255])

        Probability = hist / 6967296

        cdf = Probability.cumsum()

        num_cdf = (cdf - cdf.min()) * 255
        den_cdf = (1 - cdf.min())
        cdf_m = num_cdf / den_cdf

        cdf = cdf_m.astype('uint8')

        im2 = cdf[fl]

        self.pixel_value = np.reshape(im2, self.darray.shape)

        self.view()

    def contrast_stretching_click(self):
        arraymax = self.darray.max()
        arraymin = self.darray.min()
        # converting im1 to float

        # contrast stretching transformation
        self.pixel_value = 255 * (self.darray - arraymin) / (arraymax - arraymin)

        self.view()

    def id_low_click(self):

        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for ILPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2

        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius

        # defining the convolution function for ILPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # high frequency
                    if r > d_0:
                        H[i, j, k] = 0.0

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue
        self.view()

    def id_high_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for IHPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius

        # defining the convolution function for IHPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # low frequency
                    if 0 < r < d_0:
                        H[i, j, k] = 0.0

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()

    def bu_low_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for BLPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius

        # defining the convolution function for BLPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # high frequency
                    if r > d_0:
                        H[i, j, k] = 1 / (1 + (r / d_0) ** 2)

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()

    def bu_high_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for BHPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius
        t1 = 1  # the order of BHPF
        t2 = 2 * t1
        # defining the convolution function for BHPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # low frequency
                    if 0 < r < d_0:
                        H[i, j, k] = 1 - 1 / (1 + (r / d_0) ** t2)

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()

    def ga_low_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for GLPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius
        t1 = 2 * d_0

        # defining the convolution function for GLPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # high frequency
                    if r > d_0:
                        H[i, j, k] = math.exp(-r ** 2 / t1 ** 2)

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()

    def ga_high_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for GHPF
        H = np.ones((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Cut-off", "Value:", 30, 0, 200)
        # cut-off radius
        t1 = 2 * d_0

        # defining the convolution function for GHPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # low frequency
                    if 0 < r < d_0:
                        H[i, j, k] = 1 - math.exp(-r ** 2 / t1 ** 2)

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()

    def bandpass_click(self):
        c = fftim.fftn(self.darray)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        O = d.shape[2]
        # values in H are initialized to 1
        print(M, N, O)
        # defining the convolution function for BPF
        H = np.zeros((M, N, O))
        center1 = M / 2
        center2 = N / 2
        center3 = O / 2
        d_0, r = QInputDialog.getDouble(MainWindow, "Get Minimum", "Minimum:", 30, 0, 100)
        d_1, r = QInputDialog.getDouble(MainWindow, "Get Maximum", "Maximum:", 60, 0, 200)

        print(H)
        # defining the convolution function for BPF
        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, O):
                    r1 = (i - center1) ** 2 + (j - center2) ** 2 + (k - center3) ** 2
                    # euclidean distance from
                    # origin is computed
                    r = math.sqrt(r1)
                    # using cut-off radius to eliminate
                    # Other frequency
                    if r > d_0 and r < d_1:
                        H[i, j, k] = 1.0

        # performing the convolution
        con = d * H
        con = fftim.ifftshift(con)
        # computing the magnitude of the inverse FFT
        arrayvalue = abs(fftim.ifftn(con))
        self.pixel_value = arrayvalue / arrayvalue.max() * 255

        self.view()


    def otsu_click(self):
        thresh = threshold_otsu(self.darray)

        bool_image = self.darray > thresh
        self.pixel_value = bool_image * 255
        self.view()

    def renyi_click(self):
        thresh = Renyi1.renyi_seg_fn(self.darray, 3)
        print(thresh)
        bool_image = self.darray > thresh
        self.pixel_value = bool_image * 255
        self.view()



    def adaptive_click(self):

        array = np.zeros((self.z,self.y,self.x))

        for i in range(0, self.z):
            darray_slice = self.darray[i, :, :]
            thre_image = filters.threshold_adaptive(darray_slice, 15, method='gaussian', offset=5)
            array[i, :, :] = array[i, :, :] + thre_image

        self.pixel_value = array * 255
        self.view()

    def watershed_click(self):

        array = np.zeros((self.z, self.y, self.x))

        for i in range(0, self.z):
            darray_slice = self.darray[i, :, :]
            darray_slice = cv2.convertScaleAbs((darray_slice/darray_slice.max())*255)
            RGB, labelled = watershed1.water(darray_slice)

            watershed_image = cv2.watershed(RGB, labelled)
            array[i, :, :] = array[i, :, :] + watershed_image
        self.pixel_value = array
        self.view()

    def dilation_click(self):

        iteration, r = QInputDialog.getInt(MainWindow, "Iterations", "Value:", 1, 0, 50)

        thresh = threshold_otsu(self.darray)

        image = self.darray > thresh
        print(image)

        bool_array = scipy.ndimage.morphology.binary_dilation(image, iterations=iteration)
        self.pixel_value = bool_array * 255
        self.view()

    def erosion_click(self):

        iteration, r = QInputDialog.getInt(MainWindow, "Iterations", "Value:", 1, 0, 50)

        thresh = threshold_otsu(self.darray)

        image = self.darray > thresh
        print(image)

        bool_array = scipy.ndimage.morphology.binary_erosion(image, iterations=iteration)
        self.pixel_value = bool_array * 255
        self.view()

    def opening_click(self):

        iteration, r = QInputDialog.getInt(MainWindow, "Iterations", "Value:", 1, 0, 50)

        thresh = threshold_otsu(self.darray)

        image = self.darray > thresh
        print(image)

        bool_array = scipy.ndimage.morphology.binary_opening(image, iterations=iteration)
        self.pixel_value = bool_array * 255
        self.view()

    def closing_click(self):

        iteration, r = QInputDialog.getInt(MainWindow, "Iterations", "Value:", 1, 0, 50)

        thresh = threshold_otsu(self.darray)

        image = self.darray > thresh
        print(image)

        bool_array = scipy.ndimage.morphology.binary_closing(image, iterations=iteration)
        self.pixel_value = bool_array * 255
        self.view()

    def hit_or_miss_click(self):

        bool_array = hit_or_miss1.hit_miss(self.darray)
        print(bool_array)
        self.pixel_value = bool_array * 255
        self.view()


    def skeletonize_click(self):
        thresh = filters.threshold_otsu(self.darray)
        self.darray = self.darray > thresh

        self.pixel_value = skeletonize_3d(self.darray)
        self.view()




    def view(self):
        pixel_value = (self.pixel_value/self.pixel_value.max()) * 255
        self.after_cvuint8 = cv2.convertScaleAbs(pixel_value)

        self.image_after = self.after_cvuint8[self.i, :, :]
        print(self.i)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.image_after)))
        self.graphicsView_2.setScene(scene)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

