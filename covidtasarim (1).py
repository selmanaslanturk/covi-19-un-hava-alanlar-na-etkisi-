# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'covidtasarim.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(804, 470)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 791, 471))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(30, 20, 131, 31))
        self.pushButton.setObjectName("pushButton")
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setGeometry(QtCore.QRect(210, 20, 601, 481))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(20, 170, 111, 21))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(30, 100, 81, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(120, 100, 51, 21))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(30, 120, 81, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab)
        self.label_6.setGeometry(QtCore.QRect(120, 120, 51, 21))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(20, 250, 121, 151))
        self.label.setText("")
        self.label.setObjectName("label")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox.setGeometry(QtCore.QRect(0, 10, 341, 301))
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 40, 101, 16))
        self.label_7.setObjectName("label_7")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(240, 260, 81, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 20, 201, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(120, 50, 201, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(108, 100, 211, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(208, 140, 111, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(10, 140, 131, 20))
        self.label_9.setObjectName("label_9")
        self.comboBox_4 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_4.setEnabled(True)
        self.comboBox_4.setGeometry(QtCore.QRect(150, 220, 171, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(4, 180, 311, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.listWidget_6 = QtWidgets.QListWidget(self.tab_2)
        self.listWidget_6.setGeometry(QtCore.QRect(600, 40, 161, 151))
        self.listWidget_6.setObjectName("listWidget_6")
        self.listWidget_7 = QtWidgets.QListWidget(self.tab_2)
        self.listWidget_7.setGeometry(QtCore.QRect(600, 240, 161, 171))
        self.listWidget_7.setObjectName("listWidget_7")
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setGeometry(QtCore.QRect(440, 220, 47, 13))
        self.label_17.setObjectName("label_17")
        self.label_16 = QtWidgets.QLabel(self.tab_2)
        self.label_16.setGeometry(QtCore.QRect(670, 20, 51, 20))
        self.label_16.setObjectName("label_16")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(690, 220, 47, 13))
        self.label_18.setObjectName("label_18")
        self.listWidget_5 = QtWidgets.QListWidget(self.tab_2)
        self.listWidget_5.setGeometry(QtCore.QRect(360, 240, 211, 171))
        self.listWidget_5.setObjectName("listWidget_5")
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setGeometry(QtCore.QRect(440, 20, 47, 13))
        self.label_15.setObjectName("label_15")
        self.listWidget_8 = QtWidgets.QListWidget(self.tab_2)
        self.listWidget_8.setGeometry(QtCore.QRect(360, 40, 211, 151))
        self.listWidget_8.setObjectName("listWidget_8")
        self.label_23 = QtWidgets.QLabel(self.tab_2)
        self.label_23.setGeometry(QtCore.QRect(370, 12, 61, 21))
        self.label_23.setText("")
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.tab_2)
        self.label_24.setGeometry(QtCore.QRect(600, 20, 61, 21))
        self.label_24.setText("")
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.tab_2)
        self.label_25.setGeometry(QtCore.QRect(360, 210, 61, 21))
        self.label_25.setText("")
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.tab_2)
        self.label_26.setGeometry(QtCore.QRect(610, 210, 61, 21))
        self.label_26.setText("")
        self.label_26.setObjectName("label_26")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        self.label_11.setGeometry(QtCore.QRect(370, 190, 361, 241))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setText("")
        self.label_11.setScaledContents(True)
        self.label_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_11.setIndent(0)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        self.label_12.setGeometry(QtCore.QRect(420, 170, 251, 20))
        self.label_12.setObjectName("label_12")
        self.label_19 = QtWidgets.QLabel(self.tab_3)
        self.label_19.setGeometry(QtCore.QRect(10, 10, 51, 16))
        self.label_19.setObjectName("label_19")
        self.listWidget_9 = QtWidgets.QListWidget(self.tab_3)
        self.listWidget_9.setGeometry(QtCore.QRect(10, 250, 261, 191))
        self.listWidget_9.setObjectName("listWidget_9")
        self.listWidget_15 = QtWidgets.QListWidget(self.tab_3)
        self.listWidget_15.setGeometry(QtCore.QRect(10, 30, 261, 171))
        self.listWidget_15.setObjectName("listWidget_15")
        self.label_20 = QtWidgets.QLabel(self.tab_3)
        self.label_20.setGeometry(QtCore.QRect(20, 230, 181, 16))
        self.label_20.setObjectName("label_20")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(370, 70, 361, 21))
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.tab_4)
        self.tableWidget_2.setGeometry(QtCore.QRect(30, 90, 331, 271))
        self.tableWidget_2.setMaximumSize(QtCore.QSize(721, 271))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(3)
        self.tableWidget_2.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(2, item)
        self.label_13 = QtWidgets.QLabel(self.tab_4)
        self.label_13.setGeometry(QtCore.QRect(96, 40, 181, 20))
        self.label_13.setText("")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.tabWidget.addTab(self.tab_4, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "VER??SET?? EKLE"))
        self.label_3.setText(_translate("Dialog", "toplam veri sayisi:"))
        self.label_5.setText(_translate("Dialog", "kolon sayisi:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "1"))
        self.groupBox.setTitle(_translate("Dialog", "MODEL EG??T??M??"))
        self.label_7.setText(_translate("Dialog", "????LEM SE????N??Z"))
        self.pushButton_4.setText(_translate("Dialog", "MODEL?? E????T"))
        self.pushButton_2.setText(_translate("Dialog", "??LKELERE G??RE TAHM??N YAP"))
        self.pushButton_3.setText(_translate("Dialog", "HAVAALANLARINA G??RE TAHM??N YAP"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "0.1"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "0.2"))
        self.comboBox_2.setItemText(2, _translate("Dialog", "0.3"))
        self.comboBox_2.setItemText(3, _translate("Dialog", "0.4"))
        self.comboBox_2.setItemText(4, _translate("Dialog", "0.5"))
        self.comboBox_2.setItemText(5, _translate("Dialog", "0.6"))
        self.comboBox_2.setItemText(6, _translate("Dialog", "0.7"))
        self.comboBox_2.setItemText(7, _translate("Dialog", "0.8"))
        self.comboBox_2.setItemText(8, _translate("Dialog", "0.9"))
        self.label_9.setText(_translate("Dialog", "H-O ????in b??lme oran??"))
        self.comboBox_4.setItemText(0, _translate("Dialog", "linear_regresyon"))
        self.comboBox_4.setItemText(1, _translate("Dialog", "KNeighborsRegressor"))
        self.comboBox_4.setItemText(2, _translate("Dialog", "super_vektor_regresyonu"))
        self.comboBox_4.setItemText(3, _translate("Dialog", "karar_agaci_algoritmasi"))
        self.comboBox_4.setItemText(4, _translate("Dialog", "random_forest_algoritmasi"))
        self.pushButton_5.setText(_translate("Dialog", "TEST-TRA??N B??L"))
        self.label_17.setText(_translate("Dialog", "X_TEST"))
        self.label_16.setText(_translate("Dialog", "Y_TRA??N"))
        self.label_18.setText(_translate("Dialog", "Y_TEST"))
        self.label_15.setText(_translate("Dialog", "X_TRA??N"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "2"))
        self.label_12.setText(_translate("Dialog", "Tahmin ve Ger??ek De??erlerin Grafiksel G??sterimi"))
        self.label_19.setText(_translate("Dialog", "Tahminler"))
        self.label_20.setText(_translate("Dialog", "Ba??ar?? metrikleri"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Dialog", "3"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Algoritma"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Ba??ar??"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "Tahmin"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Dialog", "4"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

