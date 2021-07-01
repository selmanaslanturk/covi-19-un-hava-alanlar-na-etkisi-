# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:11:36 2021

@author: LENOVO
"""
import os
import csv
import pandas as pd 
from pandas import array
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from  keras.utils import np_utils 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from PyQt5.uic import loadUiType
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from covidtasarim import Ui_Dialog
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.neighbors import  KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QTimer,QTime,QAbstractTableModel
from PyQt5 import uic
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
from PyQt5.QtWidgets import QApplication, QTableView, QFileDialog,QMessageBox,QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem
from sklearn.preprocessing import LabelEncoder


class MainWindow(QWidget,Ui_Dialog):
    
    dataset_file_path = ""
    def __init__(self):
        
        QtWidgets.QMainWindow.__init__(self)  
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.verisetiekle)
        self.pushButton_2.clicked.connect(self.islemulkeyegore)
        self.pushButton_3.clicked.connect(self.islemhavaalaninagore)
        self.pushButton_5.clicked.connect(self.holdout)
        self.pushButton_4.clicked.connect(self.makine)        
    def verisetiekle(self):
        self.comboBox.clear()
        self.dosyam, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Veri Seti Seç","","Veri Seti Türü(*.csv)")
        self.veriler=pd.read_csv(self.dosyam)
        print (self.veriler)
        self.dataset=self.veriler.values
        self.data=self.dataset[:,0:self.dataset.shape[0]]     
        self.tableWidget.setRowCount(self.data.shape[0])
        self.tableWidget.setColumnCount(self.data.shape[1])
        
        for i in range(0,self.dataset.shape[0]):
           
            for j in range(0,self.dataset.shape[1]):
                self.tableWidget.setItem(i,j,QtWidgets.QTableWidgetItem(str(self.dataset[i][j])))
        
       
        self.label_4.setText(str(self.data.shape[0]))
        self.label_6.setText(str(self.data.shape[1]))    
         
         
    def islemulkeyegore(self):
        print(type(self.veriler))
        self.yeniveriler=self.veriler.iloc[5936:, 0:3]
        print(type(self.yeniveriler))
        self.dataset=self.yeniveriler.values
        self.data=self.dataset[:,0:self.dataset.shape[0]]     
        self.tableWidget.setRowCount(self.data.shape[0])
        self.tableWidget.setColumnCount(self.data.shape[1])
        print(type(self.yeniveriler))
        for i in range(0,self.dataset.shape[0]):
           
            for j in range(0,self.dataset.shape[1]):
                self.tableWidget.setItem(i,j,QtWidgets.QTableWidgetItem(str(self.dataset[i][j])))
        self.label_4.setText(str(self.data.shape[0]))
        self.label_6.setText(str(self.data.shape[1])) 
        # print(self.yeniveriler)
        self.comboBox.clear()
        self.label_8.setText("Ülke seçiniz:")
        self.comboBox.addItem("Australia")
        self.comboBox.addItem("Chile")
        self.comboBox.addItem("Canada")
        self.comboBox.addItem("United States of America (the)")
        # ulkelist=["Australia","Canada","Chile","United States of America (the)"]
      
        # self.list=ulkelist
    def islemhavaalaninagore(self):
        self.yeniveriler=self.veriler.iloc[:5936, 0:5]
        self.dataset=self.yeniveriler.values
        self.data=self.dataset[:,0:self.dataset.shape[0]]     
        self.tableWidget.setRowCount(self.data.shape[0])
        self.tableWidget.setColumnCount(self.data.shape[1])
        
        for i in range(0,self.dataset.shape[0]):
           
            for j in range(0,self.dataset.shape[1]):
                self.tableWidget.setItem(i,j,QtWidgets.QTableWidgetItem(str(self.dataset[i][j])))
        self.label_4.setText(str(self.data.shape[0]))
        self.label_6.setText(str(self.data.shape[1])) 
        print(self.yeniveriler)
        self.comboBox.clear()
        self.label_8.setText("Havaalanı seçiniz:")
        self.comboBox.addItem("Boston Logan International")
        self.comboBox.addItem("Calgary International")
        self.comboBox.addItem("Charlotte Douglas International")
        self.comboBox.addItem("Chicago OHare International")   
        self.comboBox.addItem("Dallas/Fort Worth International") 
        self.comboBox.addItem("Daniel K. Inouye International")    
        self.comboBox.addItem("Denver International")
        self.comboBox.addItem("Detroit Metropolitan Wayne County")    
        self.comboBox.addItem("Edmonton International")    
        self.comboBox.addItem("Halifax International")    
        self.comboBox.addItem("Hamilton International")  
        self.comboBox.addItem("Hartsfield-Jackson Atlanta International")  
        self.comboBox.addItem("John F. Kennedy International")                  
        self.comboBox.addItem("Kingsford Smith")
        self.comboBox.addItem("LaGuardia")
        self.comboBox.addItem("Los Angeles International")
        self.comboBox.addItem("McCarran International")
        self.comboBox.addItem("Miami International")
        self.comboBox.addItem("Montreal Mirabel")
        self.comboBox.addItem("Montreal Trudeau")
        self.comboBox.addItem("Newark Liberty International ")
        self.comboBox.addItem("San Francisco International")
        self.comboBox.addItem("Santiago International Airport")
        self.comboBox.addItem("Seattle-Tacoma International")
        self.comboBox.addItem("Toronto Pearson")
        self.comboBox.addItem("Vancouver International")
        self.comboBox.addItem("Washington Dulles International")
        self.comboBox.addItem("Winnipeg International")
       #  havaalanilist=["Boston Logan International","Calgary International","Charlotte Douglas International",
       #              "Chicago OHare International",
           
       # "Dallas/Fort Worth International",
           
       # "Daniel K. Inouye International" ,
            
       # "Denver International",
          
       # "Detroit Metropolitan Wayne County"  ,
           
       # "Edmonton International",
          
       # "Halifax International",
                         
       # "Hamilton International",
         
       # "Hartsfield-Jackson Atlanta International",
          
       # "John F. Kennedy International"    ,     
            
       # "Kingsford Smith",
           
       # "LaGuardia",
            
       # "Los Angeles International",
           
       # "McCarran International",
            
       # "Miami International",
          
       # "Montreal Mirabel",
           
       # "Montreal Trudeau",
           
       # "Newark Liberty International ",
            
       # "San Francisco International",
            
       # "Santiago International Airport",
            
       # "Seattle-Tacoma International",
           
       # "Toronto Pearson",
           
       # "Vancouver International",
           
       # "Washington Dulles International",
          
       
       # "Winnipeg International"]
       #  self.list=havaalanilist 
    def holdout(self):
      
       self.listWidget_6.clear()
       self.listWidget_7.clear()
       self.listWidget_8.clear()
       self.listWidget_5.clear()
       
       
       combo=self.comboBox.currentText()
       
       # for a in self.list:
       #  combo=a

            
       if combo=="Australia":
            y=self.yeniveriler.iloc[:211,1:2].values
            X=self.yeniveriler.iloc[:211,0:1].values
            print(str(X))
            print(y)   
       if combo=="Canada":
            y=self.yeniveriler.iloc[211:427,1:2].values
            X=self.yeniveriler.iloc[211:427,0:1].values
            print(str(X))
            print(y)   
       
       if combo=="Chile":     
            y=self.yeniveriler.iloc[427:622,1:2].values
            X=self.yeniveriler.iloc[427:622,0:1].values
            print(str(X))
            print(y)   
       if combo=="United States of America (the)":
            y=self.yeniveriler.iloc[622:,1:2].values
            X=self.yeniveriler.iloc[622:,0:1].values
            print(str(X))
            print(y)    
            
       if combo=="Boston Logan International":
            X=self.yeniveriler.iloc[:214,0:1].values
            y=self.yeniveriler.iloc[:214,1:2].values
            print(str(X))
            print(y)
       if combo=="Calgary International":
           X=self.yeniveriler.iloc[214:428,0:1].values
           y=self.yeniveriler.iloc[214:428,1:2].values
           print(str(X))
           print(y)
       if combo=="Charlotte Douglas International":
           X=self.yeniveriler.iloc[428:642,0:1].values
           y=self.yeniveriler.iloc[428:642,1:2].values
           print(str(X))
           print(y)
       if combo=="Chicago OHare International":
            y=self.yeniveriler.iloc[642:856,1:2].values
            X=self.yeniveriler.iloc[642:856,0:1].values
            print(str(X))
            print(y)
       if combo=="Dallas/Fort Worth International":
            y=self.yeniveriler.iloc[856:1071,1:2].values
            X=self.yeniveriler.iloc[856:1071,0:1].values
            print(str(X))
            print(y)
       if combo=="Daniel K. Inouye International" :
            y=self.yeniveriler.iloc[1071:1285,1:2].values
            X=self.yeniveriler.iloc[1071:1285,0:1].values
            print(str(X))
            print(y)
       if combo=="Denver International":
            y=self.yeniveriler.iloc[1285:1499,1:2].values
            X=self.yeniveriler.iloc[1285:1499,0:1].values
            print(str(X))
            print(y)
       if combo=="Detroit Metropolitan Wayne County"  :
            y=self.yeniveriler.iloc[1499:1713,1:2].values
            X=self.yeniveriler.iloc[1499:1713,0:1].values
            print(str(X))
            print(y)
       if combo=="Edmonton International":
            y=self.yeniveriler.iloc[1713:1902,1:2].values
            X=self.yeniveriler.iloc[1713:1902,0:1].values
            print(str(X))
            print(y)
       if combo=="Halifax International":
            y=self.yeniveriler.iloc[1902:2107,1:2].values
            X=self.yeniveriler.iloc[1902:2107,0:1].values
            print(str(X))
            print(y)
       if combo=="Hamilton International":
            y=self.yeniveriler.iloc[2107:2320,1:2].values
            X=self.yeniveriler.iloc[2107:2320,0:1].values
            print(str(X))
            print(y)
       if combo=="Hartsfield-Jackson Atlanta International":
            y=self.yeniveriler.iloc[2320:2535,1:2].values
            X=self.yeniveriler.iloc[2320:2535,0:1].values
            print(str(X))
            print(y)
       if combo=="John F. Kennedy International"       :     
            y=self.yeniveriler.iloc[2535:2750,1:2].values
            X=self.yeniveriler.iloc[2535:2750,0:1].values
            print(str(X))
            print(y)
       if combo=="Kingsford Smith":
            y=self.yeniveriler.iloc[2750:2961,1:2].values
            X=self.yeniveriler.iloc[2750:2961,0:1].values
            print(str(X))
            print(y)
       if combo=="LaGuardia":
            y=self.yeniveriler.iloc[2961:3175,1:2].values
            X=self.yeniveriler.iloc[2961:3175,0:1].values
            print(str(X))
            print(y)
       if combo=="Los Angeles International":
            y=self.yeniveriler.iloc[3175:3390,1:2].values
            X=self.yeniveriler.iloc[3175:3390,0:1].values
            print(str(X))
            print(y)
       if combo=="McCarran International":
            y=self.yeniveriler.iloc[3390:3604,1:2].values
            X=self.yeniveriler.iloc[3390:360,0:1].values
            print(str(X))
            print(y)
       if combo=="Miami International":
            y=self.yeniveriler.iloc[3604:3818,1:2].values
            X=self.yeniveriler.iloc[3604:3818,0:1].values
            print(str(X))
            print(y)
       if combo=="Montreal Mirabel":
            y=self.yeniveriler.iloc[3818:4028,1:2].values
            X=self.yeniveriler.iloc[3818:4028,0:1].values
            print(str(X))
            print(y)
       if combo=="Montreal Trudeau":
            y=self.yeniveriler.iloc[4028:4243,1:2].values
            X=self.yeniveriler.iloc[4028:4243,0:1].values
            print(str(X))
            print(y)
       if combo=="Newark Liberty International ":
            y=self.yeniveriler.iloc[4243:4457,1:2].values
            X=self.yeniveriler.iloc[4243:4457,0:1].values
            print(str(X))
            print(y)
       if combo=="San Francisco International":
            y=self.yeniveriler.iloc[4457:4671,1:2].values
            X=self.yeniveriler.iloc[4457:4671,0:1].values
            print(str(X))
            print(y)
       if combo=="Santiago International Airport":
            y=self.yeniveriler.iloc[4671:4866,1:2].values
            X=self.yeniveriler.iloc[4671:4866,0:1].values
            print(str(X))
            print(y)
       if combo=="Seattle-Tacoma International":
            y=self.yeniveriler.iloc[4866:5080,1:2].values
            X=self.yeniveriler.iloc[4866:5080,0:1].values
            print(str(X))
            print(y)
       if combo=="Toronto Pearson":
            y=self.yeniveriler.iloc[5080:5294,1:2].values
            X=self.yeniveriler.iloc[5080:5294,0:1].values
            print(str(X))
            print(y)
       if combo=="Vancouver International": 
            y=self.yeniveriler.iloc[5294:5508,1:2].values
            X=self.yeniveriler.iloc[5294:5508,0:1].values
            print(str(X))
            print(y)
       if combo=="Washington Dulles International":
            y=self.yeniveriler.iloc[5508:5722,1:2].values
            X=self.yeniveriler.iloc[5508:5722,0:1].values
            print(str(X))
            print(y)
       if combo=="Winnipeg International":
            y=self.yeniveriler.iloc[5722:,1:2].values
            X=self.yeniveriler.iloc[5722:,0:1].values
            print(str(X))
            print(y)
       lb = LabelEncoder()      
       X[:,0] = lb.fit_transform(X[:,0])    
       # from sklearn import preprocessing
       # ohe= preprocessing.OneHotEncoder()
      
       # X =ohe.fit_transform(X).toarray()

       # X = pd.DataFrame(data=X[:,:1],index=range(len(X)),columns=['Date',])


       # X = lb.fit_transform(X)
       print(X)
       # X=X.reshape(-1,1)
       self.X_train,  self.X_test,  self.Y_train,  self.Y_test = train_test_split(X, y, test_size =float(self.comboBox_2.currentText()))
 
       sc=StandardScaler()

       self.X_train = sc.fit_transform(self.X_train)
       self.X_test = sc.fit_transform(self.X_test)

  
       self.label_23.setText(str(self.X_train.shape))
       self.label_24.setText(str(self.Y_train.shape))  
       self.label_25.setText(str(self.X_test.shape))  
       self.label_26.setText(str(self.Y_test.shape))  
       self.listWidget_8.addItem(str(self.X_train))
       self.listWidget_7.addItem(str(self.Y_test))
       self.listWidget_5.addItem(str(self.X_test))
       self.listWidget_6.addItem(str(self.Y_train))
      
    def makine(self):
        self.pandas_list=[]
        self.basarilist=[]
        self.ortmutlakhatalist=[]
        self.ortkarehatalist=[]
        self.kokortlist=[]
        self.ortalama=[]
        
        self.listWidget_15.clear()
        # self.listWidget_9.clear()
        mkalist=["linear_regresyon",
          "super_vektor_regresyonu", "karar_agaci_algoritmasi",
            "random_forest_algoritmasi","KNeighborsRegressor"]
        for i in mkalist:
         ort=0
         # mk=self.comboBox_4.currentText()
         mk=i
         if mk=="linear_regresyon":
           self.mka=LinearRegression()
         if mk=="super_vektor_regresyonu": 
           self.mka= SVR ()
         if mk=="karar_agaci_algoritmasi": 
            self.mka=DecisionTreeRegressor()
         if mk=="random_forest_algoritmasi":
            self.mka=RandomForestRegressor()    
         if mk=="KNeighborsRegressor":
            self.mka=KNeighborsRegressor()     
        
       
            
              
         self.mka.fit(self.X_train, self.Y_train)
         self.y_pred= self.mka.predict(self.X_test)
         a=0
         for i in self.y_pred:
          a+=1  
          ort+=i
          
         ort=int(ort/a)
         self.ortalama.append(ort)
         print()  
         self.label_10.setText("Bir Sonraki Gün İçin Beklenen Tahmini Uçuş Sayısı: {}".format(ort)) 
         self.listWidget_15.addItem(str(self.y_pred))
         self.grafik()
         self.regbasari()
        # self.basarilist=np.array(self.basarilist)
        # pandas_series = pd.Series(self.basarilist ,columns=['Başarı'])
        # pandas_series = pd.Series(self.ortmutlakhatalist ,columns=[' ortalama mutlak hata değeri '])
        # pandas_series = pd.Series(self.ortkarehatalist ,columns=['ortalama karesel hata değeri  '])
        # pandas_series = pd.Series(self.kokortlist ,columns=['kök ortalama karesel hata değeri'])
        print(self.ortalama)
        for i in range(len(self.basarilist)):
            
                
            self.pandas_list.append([mkalist[i],self.basarilist[i],self.ortalama[i]])
        df =pd.DataFrame (self.pandas_list,columns=['Algoritma','Başarı','Tahmini Uçuş Sayısı'])
    
      
        combo=self.comboBox.currentText()

        print(df) 
        # self.label_13.setText(str(df))
        self.label_13.setText(str(combo))
       
        self.datset=df.values
        self.dat=self.datset[:,0:self.datset.shape[0]]     
        self.tableWidget_2.setRowCount(self.dat.shape[0])
        self.tableWidget_2.setColumnCount(self.dat.shape[1])
        self.tableWidget_2.columns=['Algoritma','Başarı','Tahmini Uçuş Sayısı']
        
        for i in range(0,self.datset.shape[0]):
           
            for j in range(0,self.datset.shape[1]):
                self.tableWidget_2.setItem(i,j,QtWidgets.QTableWidgetItem(str(self.datset[i][j])))
        
    def grafik(self):
        
        plt.plot(self.y_pred)
        plt.plot(self.Y_test)
        # plt.plot(self.X_train,self.Y_train)
        # plt.plot(self.X_test,self.y_pred)
        plt.title('tahmin gerçek')
      
        plt.legend(['tahmin', 'gerçek'], loc='lower right')
        
        plt.savefig("./plot.png")
        self.pixmap = QPixmap("./plot.png")
        self.label_11.setPixmap(self.pixmap)
        plt.show()         
    def regbasari(self):
        
        mae = mean_absolute_error(self.Y_test, self.y_pred) #ortalama karesel hata
        mse = mean_squared_error(self.Y_test,self.y_pred, squared=True)
        rmse = mean_squared_error(self.Y_test,self.y_pred, squared=False)
        r2 = r2_score(self.Y_test,self.y_pred)
        self.basarii=r2
        self.listWidget_9.addItem(" Başarı(R kare skoru) değeri      : {:.2%}" .format (r2))
        self.listWidget_9.addItem(" ortalama mutlak hata değeri      : {:.2%}" .format (mae))
        self.listWidget_9.addItem(" ortalama karesel hata değeri     : {:.2%}" .format (mse))
        self.listWidget_9.addItem(" kök ortalama karesel hata değeri : {:.2%}" .format (rmse))
        
        self.basarilist.append("{:.2%}" .format (r2))
        self.ortmutlakhatalist.append(mae)
        self.ortkarehatalist.append(mse)
        self.kokortlist.append(rmse)
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
# from covidproje import MainWindow

def main():
 app = QtWidgets.QApplication(sys.argv)
 mainWindow = MainWindow()
 mainWindow.show()
 sys.exit(app.exec_())



if __name__ == "__main__":

  main()   