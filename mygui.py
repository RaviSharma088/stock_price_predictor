from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import pandas as pd
import numpy as np
import preprocess_data as ppd
import lstm, time #helper libraries
import visualize as vs
import lstm
import stock_data as sd
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18, 12)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pandas import DataFrame
# D://Ravi Sharma//finalproject//final code


class res:
    def __init__(self):
        # creating window 
        self.master=Tk()
        self.master.title("STOCK PRICE PREDICTION")
        self.master.configure(background="white")  
        self.master.geometry("700x600")
        #creating frames for inserting image ,labels and buttons on window 
        self.p1=ImageTk.PhotoImage(Image.open("D://Ravi Sharma//finalproject//final code//1.JPG"))
        self.f1=Frame(self.master, bg="#F9CEEE", relief = RAISED).place(x=0, y=0, width=700, height=600)
        self.l2=Label(self.f1,image=self.p1).place(x=0,y=0,width=700,height=500)
        self.b1=Button(self.f1, text="Press to start", font = "Helvetica 15 bold italic ",borderwidth=0,activebackground="#F9CEEE",command=self.start,fg="black",bg="#F9CEEE" )
        self.b1.place(x=430,y=525,width=200,height=50)
   

    
# this is the 2nd page of the window where we select company data set       
    def start(self):
        
        self.p2=ImageTk.PhotoImage(Image.open("D://Ravi Sharma//finalproject//final code//2.JPG"))
        self.f2=Frame(self.master, bg="#F9CEEE", relief = RAISED).place(x=0, y=0, width=700, height=150)
        self.f3=Frame(self.master, bg="#F9CEEE", relief = RAISED).place(x=0, y=150, width=700, height=450)
        self.l3=Label(self.f2,image=self.p2).place(x=0,y=0,width=700,height=150)
        self.l3=Label(self.f3, text="Choose the company",fg="Black", font = "Helvetica 13 italic", bg="#F9CEEE")
        self.l3.place(x=80,y=150,width=200,height=50)
        self.b2=Button(self.f3,text="Select", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="#F9CEEE",activebackground="#F9CEEE",command=self.dataset)
        self.b2.place(x=350,y=350,width=200,height=50)
        self.b3=Button(self.f3,text="Next", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="#F9CEEE",activebackground="#F9CEEE",command=self.algo)
        self.b3.place(x=150,y=350,width=200,height=50)
        #creating combo box
        self.n=tk.StringVar()
        self.companyname = ttk.Combobox(self.f3,textvariable=self.n)
        self.companyname['values']=('Amazon','Google','Microsoft','Apple','TCS','Tesla','Accenture','SBI')
        self.companyname.place(x=400,y=160,width=100,height=30) 
        self.companyname.current()
        self.companyname.bind("<<ComboboxSelected>>",self.callbackFunc)

    



# this func read value from combobox to choose data set according to choosen company
 
    def callbackFunc(self,event):
        self.st=self.companyname.get()
        
        
 
    
 # this func show which dataset you have choosen on GUI 2nd page   
    def dataset(self):
        self.l4=Label(self.f3, font = "Helvetica 13 italic", bg="#F9CEEE")
        self.l4.place(x=25,y=205,width=300,height=100)
        if self.st == "Amazon":
            self.data=pd.read_csv("Amazon_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "Google":
            self.data=pd.read_csv("Google_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "Microsoft":
            self.data=pd.read_csv("Microsoft_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "TCS":
            self.data=pd.read_csv("tcs_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "Tesla":
            self.data=pd.read_csv("Tesla_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "Accenture":
            self.data=pd.read_csv("ACN_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        elif self.st == "SBI":
            self.data=pd.read_csv("SBI_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        else:
            self.data=pd.read_csv("Apple_preprocessed.csv")
            self.l4.config(text=self.st+" is selected")
            print(self.data.head())
        return self.data




# this is 3rd page of GUI where we choose algo
    def algo(self):
        self.p2=ImageTk.PhotoImage(Image.open("D://Ravi Sharma//finalproject//final code//1.JPG"))
        self.f2=Frame(self.master, bg="#F9CEEE", relief = RAISED).place(x=0, y=0, width=700, height=150)
        self.f4=Frame(self.master, bg="#F9CEEE", relief = RAISED).place(x=0, y=150, width=700, height=450)
        self.l3=Label(self.f2,image=self.p2).place(x=0,y=0,width=700,height=150)
        self.l3=Label(self.f4, text="Choose the Algorithm",fg="Black", font = "Helvetica 13 italic", bg="#F9CEEE")
        self.l3.place(x=80,y=150,width=200,height=50)
        self.b4=Button(self.f4,text="Next", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="#F9CEEE",activebackground="#F9CEEE",command=self.runalgo)
        self.b4.place(x=350,y=350,width=200,height=50)
        #self.b5=Button(self.f4,text="graph", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="grey",activebackground="grey",command=self.graph)
        #self.b5.place(x=150,y=350,width=200,height=50)
        self.n=tk.StringVar()
        self.algoname = ttk.Combobox(self.f4,textvariable=self.n)
        self.algoname['values']=('Support Vector Machine','Neural Network')
        self.algoname.place(x=400,y=160,width=200,height=30) 
        self.algoname.current()
        self.algoname.bind("<<ComboboxSelected>>",self.choosealgo)      
 
    


# same as abovecall back func .it read algo name     
    def choosealgo(self,event):
        self.st2=self.algoname.get()
       


# this func will call choosen algo     
    def runalgo(self):
        self.l4=Label(self.f4, font = "Helvetica 13 italic", bg="grey")
        self.l4.place(x=25,y=205,width=300,height=100)
        if self.st2 == "Support Vector Machine":
            self.p,self.a=self.SVM(self.data) 
            self.l4.config(text=self.st2+" is selected")
            self.graph()
        if self.st2 == "Neural Network":
            self.a,self.p=self.neuralnetwork(self.data)
            self.l4.config(text=self.st2+" has finished its wrok")
            self.graph()
            
    
    
    
    
    
    def neuralnetwork(self,stocks):
        stocks_data = stocks.drop(['Item'], axis = 1)

        print(stocks_data.head())

        #Split train and test data sets and Unroll train and test data for improved lstm model
        X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)

        unroll_length = 50
        X_train = sd.unroll(X_train, unroll_length)
        X_test = sd.unroll(X_test, unroll_length)
        y_train = y_train[-X_train.shape[0]:]
        y_test = y_test[-X_test.shape[0]:]
        
        print("x_train", X_train.shape)
        print("y_train", y_train.shape)
        print("x_test", X_test.shape)
        print("y_test", y_test.shape)


        # Set up hyperparameters
        batch_size = 512
        epochs = 20

        # build improved lstm model
        model = lstm.build_improved_model( X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

        start = time.time()
        #final_model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('compilation time : ', time.time() - start)

        #Train improved LSTM model

        model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.05)

        #Make prediction on improved LSTM model

        predictions = model.predict(X_test, batch_size=batch_size)
        
        self.values=predictions
        #plot the results

        vs.plot_lstm_prediction(predictions, y_test)
        return predictions,y_test
   
    
   
    
   
    
   
    def SVM(self,dataset):
    
        X = dataset.iloc[:, 1:2].values
        y = dataset.iloc[:, 2:3].values
    
        # Splitting the dataset into the Training set and Test set
        #from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        #from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        y_train = sc_y.fit_transform(y_train)
        

        print("x_train", X_train.shape)
        print("y_train", y_train.shape)
        print("x_test", X_test.shape)
        print("y_test", y_test.shape)
        
        # Fitting SVR to the dataset
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        
        # Predicting a new result
        y_pred = regressor.predict(y_test)
        self.values=y_pred
        y_pred = sc_y.inverse_transform(y_pred)
        
        


        #Graph plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Add labels
        plt.ylabel("Price USD")
        plt.xlabel("Trading Days")

        # Plot actual and predicted close values
        
        plt.plot(y_test, '#00FF00', label='Predicted Close')
        plt.plot(y_pred, '#0000FF', label='Actual Close')
        
        # Set title
        ax.set_title("Trading vs Prediction")
        ax.legend(loc='upper left')


        plt.show()
        return y_test,y_pred    
        
    
    
    
    
    def graph(self):
        self.f5=Frame(self.master, bg="white", relief = RAISED).place(x=0, y=150, width=700, height=550)
        
        self.b6=Button(self.f5,text="Next", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="white",activebackground="white",command=self.results)
        self.b6.place(x=600,y=170,width=80,height=30)       
        
        df1 = DataFrame(self.a) 
        df2 = DataFrame(self.p) 
        
        figure1 = plt.Figure(figsize=(6,5), dpi=100)
        ax1 = figure1.add_subplot(111)
        ax2 = figure1.add_subplot(111)
      

        bar1 = FigureCanvasTkAgg(figure1, self.f5)
        bar1.get_tk_widget().place(x=0,y=200,height=350,width=700)
        
        df1.plot( legend=True, ax=ax1,label="adjusted close")
        df2.plot( legend=True, ax=ax2,label="predicted close")
        
        
        ax1.set_title('Trading vs Prediction')
        ax1.set_ylabel("Price USD")
        ax1.set_xlabel("trading days")
    
    
    
    
    
    def results(self):
        self.f6=Frame(self.master, bg="white", relief = RAISED).place(x=0, y=150, width=700, height=550)
        self.b6=Button(self.f6,text="Predict again", borderwidth=0,font = "Helvetica 25 bold italic",fg="black",bg="white",activebackground="white",command=self.start)
        self.b6.place(x=400,y=450,width=220,height=50) 
        if self.st2 == "Support Vector Machine":
             self.l4=Label(self.f6, text="Predicted value by Support vector machine",fg="Black", font = "Helvetica 13 italic", bg="white")
        if self.st2 == "Neural Network":
             self.l4=Label(self.f6, text="Predicted value by neural network",fg="Black", font = "Helvetica 13 italic", bg="white")
        
        self.l3=Label(self.f6, text=+self.values[-1],fg="Black", font = "Helvetica 13 italic", bg="white")
        self.l3.place(x=460,y=200,width=240,height=40)
        self.l4.place(x=20,y=200,width=400,height=40)
        
        
    
m=res()
mainloop()
