import tkinter as tk
from tkinter.filedialog import askopenfilename
import os
#import sys
from PIL import Image , ImageTk 
import cv2
import csv
import pandas as pd
root = tk.Tk()

FName=tk.StringVar()
FName.set("")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("leaf diease Detection")
root.configure(background ="lightblue")

title = tk.Label(text="Sorghum Leaf Diease Detection", background = "lightblue", fg="red", font=("", 30))
#title.pack()
title.grid(column=1, row=0 ,sticky=tk.NSEW)

def Clear_data():
    global Sel_F
    global Col_1
    global Col_2
    global Col_3
    global Col_4
    global Col_5
    global Col_6
    global Col_7
    
    Sel_F=''
    Col_1=''
    Col_2=''
    Col_3=''
    Col_4=''
    Col_5=0
    Col_6=0
    Col_7=0
    

def get_data():

    global Sel_F
    global Col_1
    global Col_2
    global Col_3
    global Col_4
    global Col_5
    global Col_6
    global Col_7

#    df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]

    imPath='E:/Alka_python/Breast_Cancer_From_MIAS/dataset/MIAS_D.csv'
#    Col_Names=['A','B','C']
    Col_Names=['Img_Name', 'Back_Tissue','Abnormality', 'B_M','X_Ab','Y_Ab','R_Ab']
#    df = pd.read_csv(imPath,header=0,names=Col_Names,engine='python',error_bad_lines=False)
    df = pd.read_csv(imPath,names=Col_Names)

    df_Val=df.loc[(df['Img_Name'] == Sel_F)]
#    df['X_Ab','Y_Ab','R_Ab'] = df['X_Ab','Y_Ab','R_Ab'].fillna(0,inplace='True')
    df_Val['X_Ab'] = df_Val['X_Ab'].fillna(0) #,inplace='True')
    df_Val['Y_Ab'] = df_Val['Y_Ab'].fillna(0)
    df_Val['R_Ab'] = df_Val['R_Ab'].fillna(0)

#    print(df_Val['X_Ab'])

    Col_1=df_Val['Img_Name'].values
    Col_2=df_Val['Back_Tissue'].values
    Col_3=df_Val['Abnormality'].values
    Col_4=df_Val['B_M'].values
    Col_5=df_Val['X_Ab'].values
    Col_6=df_Val['Y_Ab'].values
    Col_7=df_Val['R_Ab'].values

    print(Col_1,Col_2,Col_3,Col_4,Col_5,Col_6,Col_7)

def openphoto():
    Clear_data()
    
 
    fileName = askopenfilename(initialdir='/dataset', title='Select image for analysis ', filetypes=[("all files","*.*")])

    imgpath=fileName 
    fn=fileName
    Sel_F=fileName.split('/').pop()
    Sel_F=Sel_F.split('.').pop(0)
    
    
    gs = cv2.cvtColor(cv2.imread(imgpath,1),cv2.COLOR_RGB2GRAY)
    x1 = int(gs.shape[0])
    y1 =int(gs.shape[1])
    
    gs = cv2.resize(gs, (x1,y1))
        
    retval, threshold = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    img = tk.Label(image=imgtk, height=x1, width=y1)
    img.image =imgtk
    img.grid(row=4, column=0,sticky=tk.NE) #, columnspan=2, rowspan=2,sticky=tk.E)   #, padx=10, pady=10)
    
  
    
def analysis():
    
    global fn
    FName=fn
    
    imgpath=FName
    
    gs = cv2.cvtColor(cv2.imread(imgpath,1),cv2.COLOR_RGB2GRAY)
    
    x1 = int(gs.shape[0])
    y1 =int(gs.shape[1])
    
    gs = cv2.resize(gs, (x1,y1))
    
    retval, threshold = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im) 
  
    img2 = tk.Label(gi.frame1, image=imgtk, height=x1, width=y1)
    img2.image =imgtk
    img2.grid(column=1, row=4,sticky=tk.NE)
    
 
def dect():

    global fn
    FName = fn

    imgpath = FName

    gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)

    x1 = int(gs.shape[0])
    y1 = int(gs.shape[1])

    gs = cv2.resize(gs, (x1, y1))

    retval, edges= cv2.edges(gs, 0, 255, cv2.CANNY_THRESH_1 + cv2. CANNY_THRESH_2)
    im = Image.fromarray(edges)
    imgtk = ImageTk.PhotoImage(image=im)

    img2 = tk.Label(gi.frame1, image=imgtk, height=x1, width=y1)
    img2.image = imgtk
    img2.grid(column=1, row=4, sticky=tk.NE)


def exit_prog():
    root.destroy()
    



root.mainloop()
