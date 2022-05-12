from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.QtGui import QPixmap
import os
import sys
import cv2
import numpy as np
import torch
from lic360_operator import Viewport

def convert_img(raw):
        cv_img = raw.to('cpu').detach().numpy().transpose(1,2,0).astype(np.uint8)
        return cv_img

def convert_cv_qt(raw):
        cv_img = raw.to('cpu').detach().numpy().transpose(1,2,0).astype(np.uint8)
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(766, 473)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 20, 120, 21))
        self.label.setObjectName("label")
        self.label2 = QtWidgets.QLabel(Dialog)
        self.label2.setGeometry(QtCore.QRect(30, 50, 120, 21))
        self.label2.setObjectName("label2")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(150, 20, 500, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit2.setGeometry(QtCore.QRect(150, 50, 500, 20))
        self.lineEdit2.setObjectName("lineEdit2")
        self.label_theta = QtWidgets.QLabel(Dialog)
        self.label_theta.setGeometry(QtCore.QRect(30, 80, 60, 21))
        self.label_theta.setObjectName("label_theta")
        self.lineEdit_theta = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_theta.setGeometry(QtCore.QRect(100, 80, 80, 20))
        self.lineEdit_theta.setObjectName("lineEdit_theta")
        self.label_phi = QtWidgets.QLabel(Dialog)
        self.label_phi.setGeometry(QtCore.QRect(280, 80, 60, 21))
        self.label_phi.setObjectName("label_phi")
        self.lineEdit_phi = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_phi.setGeometry(QtCore.QRect(340, 80, 80, 20))
        self.lineEdit_phi.setObjectName("lineEdit_pi")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(20, 160, 300, 200))
        self.graphicsView.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView2.setGeometry(QtCore.QRect(420, 160, 300, 200))
        self.graphicsView2.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.graphicsView2.setObjectName("graphicsView2")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(650, 80, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.fButton = QtWidgets.QPushButton(Dialog)
        self.fButton.setGeometry(QtCore.QRect(680, 20, 75, 20))
        self.fButton.setObjectName("fButton")
        self.fButton2 = QtWidgets.QPushButton(Dialog)
        self.fButton2.setGeometry(QtCore.QRect(680, 50, 75, 20))
        self.fButton2.setObjectName("fButton")
        self.slabel = QtWidgets.QLabel(Dialog)
        self.slabel.setGeometry(QtCore.QRect(30, 380, 120, 21))
        self.slabel.setObjectName("slabel")
        self.slineEdit = QtWidgets.QLineEdit(Dialog)
        self.slineEdit.setGeometry(QtCore.QRect(150, 380, 500, 20))
        self.slineEdit.setObjectName("slineEdit")
        self.sButton = QtWidgets.QPushButton(Dialog)
        self.sButton.setGeometry(QtCore.QRect(660, 380, 90, 20))
        self.sButton.setObjectName("sButton")
        self.hlabel = QtWidgets.QLabel(Dialog)
        self.hlabel.setGeometry(QtCore.QRect(30, 120, 650, 21))
        self.hlabel.setObjectName("slabel")
        self.rtlabel = QtWidgets.QLabel(Dialog)
        self.rtlabel.setGeometry(QtCore.QRect(110, 140, 100, 21))
        self.rtlabel.setObjectName("rtlabel")
        self.ttlabel = QtWidgets.QLabel(Dialog)
        self.ttlabel.setGeometry(QtCore.QRect(520, 140, 100, 21))
        self.ttlabel.setObjectName("ttlabel")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "360 Image Viewer"))
        self.label.setText(_translate("Dialog", "Reference Image Path"))
        self.label2.setText(_translate("Dialog", "Compare Image Path"))
        self.label_theta.setText(_translate("Dialog", "Theta (Lat)"))
        self.label_phi.setText(_translate("Dialog", "Phi (Lon)"))
        self.lineEdit_phi.setText(_translate("Dialog", "0"))
        self.lineEdit_theta.setText(_translate("Dialog", "0"))
        self.lineEdit.setText(_translate("Dialog", "E:/360_dataset/model/ssim/img/raw/0_00179_bpg.png"))
        self.lineEdit2.setText(_translate("Dialog", "E:/360_dataset/model/ssim/img/raw/0_00163_our_ssim.png"))
        self.pushButton.setText(_translate("Dialog", "Start"))
        self.fButton.setText(_translate("Dialog", "Load file"))
        self.fButton2.setText(_translate("Dialog", "Load file"))
        self.slabel.setText(_translate("Dialog", "Output Path (Viewport)"))
        self.sButton.setText(_translate("Dialog", "Save Viewport"))
        self.slineEdit.setText(_translate("Dialog", "E:/"))
        self.rtlabel.setText(_translate("Dialog", "Reference Viewport"))
        self.ttlabel.setText(_translate("Dialog", "Target Viewport"))
        self.hlabel.setText(_translate("Dialog", 'Clip start to view the viewports. Navigation: key "a"("s") minus(plus) Theta for 0.5 degree;  key "d"("f") minus(plus) Phi for 0.5 degree;'))

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.checkPath)
        self.ui.fButton.clicked.connect(self.load_file)
        self.ui.fButton2.clicked.connect(self.load_file2)
        self.ui.lineEdit_theta.textChanged.connect(self.on_theta_changed)
        self.ui.lineEdit_phi.textChanged.connect(self.on_phi_changed)
        self.ui.sButton.clicked.connect(self.save_vp)
        self.vp = Viewport(90,171,256,device=0)
        self.imgs = None

    def save_vp(self):
        if not self.imgs is None:
            theta = float(self.ui.lineEdit_theta.text())
            phi = float(self.ui.lineEdit_phi.text())
            tp_vec = torch.Tensor([[phi,theta],[phi,theta]]).type(torch.float32).to('cuda:0').contiguous()
            y = self.vp(self.imgs,tp_vec)
            y = torch.clip(y,0,255)
            ref,tar = convert_img(y[0]),convert_img(y[1])
            pd = self.ui.slineEdit.text()
            cv2.imwrite('{}/ref_vp.png'.format(pd),ref)
            cv2.imwrite('{}/tar_vp.png'.format(pd),tar)

    def load_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'E:/360_dataset/model/param/res/ssim_vs/imgs',"Image files (*.jpg *.gif *.png)")
        self.ui.lineEdit.setText(fname[0])

    def load_file2(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'E:/360_dataset/model/param/res/ssim_vs/imgs',"Image files (*.jpg *.gif *.png)")
        self.ui.lineEdit2.setText(fname[0])
    
    def on_theta_changed(self):
        if isfloat(self.ui.lineEdit_theta.text()):
            val = float(self.ui.lineEdit_theta.text())
            tval = val + 0
            if val < -90: val = -90
            if val > 90: val = 90
            if not tval == val:
                self.ui.lineEdit_theta.setText(str(val))
        else:
            self.ui.lineEdit_theta.setText("0")
        self.move_vp()
    
    def on_phi_changed(self):
        if isfloat(self.ui.lineEdit_phi.text()):
            val = float(self.ui.lineEdit_phi.text())
            tval = val + 0
            if val < -180: val = -180
            if val > 180: val = 180
            if not tval == val:
                self.ui.lineEdit_phi.setText(str(val))
        else:
            self.ui.lineEdit_phi.setText("0")
        self.move_vp()
            

    def prepare_data(self,ref,tar):
        rimg = cv2.imread(ref)
        timg = cv2.imread(tar)
        tmp = np.array([rimg,timg]).transpose(0,3,1,2).astype(np.float32)
        self.imgs = torch.from_numpy(tmp).to('cuda:0').contiguous()

    def get_vp(self):
        theta = float(self.ui.lineEdit_theta.text())
        phi = float(self.ui.lineEdit_phi.text())
        tp_vec = torch.Tensor([[phi,theta],[phi,theta]]).type(torch.float32).to('cuda:0').contiguous()
        y = self.vp(self.imgs,tp_vec)
        y = torch.clip(y,0,255)
        return convert_cv_qt(y[0]),convert_cv_qt(y[1])
    
    def keyPressEvent(self, event):
        # keyPressEvent defined in child
        #print('pressed from myDialog: ', event.key())
        if event.key() == 65:#key a
            theta =  float(self.ui.lineEdit_theta.text()) - 0.5
            self.ui.lineEdit_theta.setText(str(theta))
        elif event.key() == 83:#key s
            theta =  float(self.ui.lineEdit_theta.text()) + 0.5
            self.ui.lineEdit_theta.setText(str(theta))
        elif event.key() == 68:#key d
            phi =  float(self.ui.lineEdit_phi.text()) - 0.5
            self.ui.lineEdit_phi.setText(str(phi))
        elif event.key() == 70:#key f
            phi =  float(self.ui.lineEdit_phi.text()) + 0.5
            self.ui.lineEdit_phi.setText(str(phi))

    def move_vp(self):
        p1,p2 = self.get_vp()
        scene1 = QtWidgets.QGraphicsScene(self)
        item1 = QtWidgets.QGraphicsPixmapItem(p1)
        scene1.addItem(item1)
        self.ui.graphicsView.setScene(scene1)
        scene2 = QtWidgets.QGraphicsScene(self)
        item2 = QtWidgets.QGraphicsPixmapItem(p2)
        scene2.addItem(item2)
        self.ui.graphicsView2.setScene(scene2)

    def checkPath(self):
        ref_path = self.ui.lineEdit.text()
        tar_path = self.ui.lineEdit2.text()
        if os.path.isfile(ref_path) and os.path.isfile(tar_path):
            self.prepare_data(ref_path,tar_path)
            p1,p2 = self.get_vp()
            scene1 = QtWidgets.QGraphicsScene(self)
            item1 = QtWidgets.QGraphicsPixmapItem(p1)
            scene1.addItem(item1)
            self.ui.graphicsView.setScene(scene1)
            scene2 = QtWidgets.QGraphicsScene(self)
            item2 = QtWidgets.QGraphicsPixmapItem(p2)
            scene2.addItem(item2)
            self.ui.graphicsView2.setScene(scene2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    class_instance = My_Application()
    class_instance.show()
    sys.exit(app.exec_())