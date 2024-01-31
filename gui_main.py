import os
import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter.filedialog import askopenfilename
import PIL
from PIL import Image, ImageTk
import cv2
import numpy as np
root = tk.Tk()
root.geometry("1300x700")

tabControl = ttk.Notebook(root)          # Create Tab Control
tab1 = ttk.Frame(tabControl)            # Create a tab



tabControl.add(tab1, text='  Pre-Process   ') # Add the tab

#tab2 = ttk.Frame(tabControl)            # Create a tab
#tabControl.add(tab2, text='   Alexnet   ')

tab3 = ttk.Frame(tabControl)            # Create a tab
tabControl.add(tab3, text='   Detection   ')

tabControl.pack(expand=True, fill="both")


#def Preprocess():
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os
# import sys
from PIL import Image, ImageTk
import cv2
import csv
import pandas as pd

FName = tk.StringVar()
FName.set("")


    # title.pack()
   # title.grid(column=1, row=0, sticky=tk.NSEW)





def bphoto():

        global fn
        global im1
        fileName = askopenfilename(initialdir='/dataset', title='Select image for analysis ',
                                   filetypes=[("all files", "*.*")])

        imgpath = fileName
        fn = fileName
        Sel_F = fileName.split('/').pop()
        Sel_F = Sel_F.split('.').pop(0)

        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
        x1 = int(gs.shape[0])
        y1 = int(gs.shape[1])

        gs = cv2.resize(gs, (x1, y1))

        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        im = Image.fromarray(gs)
        imgtk = ImageTk.PhotoImage(image=im)
        img = tk.Label(frame1, image=imgtk, height=x1, width=y1)


        img.image = imgtk
        img.grid(row=4, column=0, sticky=tk.NE)  # , columnspan=2, rowspan=2,sticky=tk.E)   #, padx=10, pady=10)


        im1 = imgtk

def analysis():
        global fn
        FName = fn

        imgpath = FName

        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)

        x1 = int(gs.shape[0])
        y1 = int(gs.shape[1])

        gs = cv2.resize(gs, (x1, y1))


        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im = Image.fromarray(threshold)
        imgtk = ImageTk.PhotoImage(image=im)

        img2 = tk.Label(frame1, image=imgtk, height=x1, width=y1)

        img2.image = imgtk
        img2.grid(column=1, row=4, sticky=tk.NE)

        global im2
        im2 = imgtk

def edges():
    global fn
    FName = fn

    global eg

    imgpath = FName


    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10


    img = cv2.imread(imgpath,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1 = int(gray.shape[0])
    y1 = int(gray.shape[1])


    gray = cv2.resize(gray, (x1, y1))
    #cv2.imshow("img", img)
    #cv2.waitKey()

    # edges
    #cv2.destroyAllWindows()
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #display image
    im = Image.fromarray(edges)
    imgtk = ImageTk.PhotoImage(image=im)
    img3 = tk.Label(frame1, image=imgtk, height=x1, width=y1)
    img3.image = imgtk
    img3.grid(column=2, row=4, sticky=tk.NE)

    eg = edges

    global im3
    im3 = imgtk


def mask():

    BLUR = 21
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10


    #import edges o/p as i/p to mask
    global eg
    edges = eg

    #import orignal image
    global fn
    FName = fn
    imgpath = FName

    global ms
    global imgm

    img = cv2.imread(imgpath, 1)

    x1 = int(edges.shape[0])
    y1 = int(edges.shape[1])

    edges = cv2.resize(edges, (x1, y1))

    contour_info = []
    contours, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c),))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    max_contour = contour_info[0]
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask] * 3)
    mask_stack = mask_stack.astype('float32') / 255.0
    imgmk = img.astype('float32') / 255.0

    im = Image.fromarray(mask)
    imgtk = ImageTk.PhotoImage(image=im)
    img4 = tk.Label(frame1, image=imgtk, height=x1, width=y1)
    img4.image = imgtk
    img4.grid(column=0, row=5, sticky=tk.N)

    ms = mask_stack
    imgm = imgmk

    global im4
    im4 = imgtk

def masked():

    global ms
    global imgm

    mask_stack = ms
    img = imgm

    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


    MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    im = Image.fromarray(masked)
    imgtk = ImageTk.PhotoImage(image=im)
    img4 = tk.Label(frame1, image=imgtk)
    img4.image = imgtk
    img4.grid(column=1, row=5, sticky=tk.W)

    global im5
    im5= imgtk

    # button4 = tk.Button(tab1, text="Refresh", command=clearAllImg, height=1, width=12, font=('times', 15, ' bold '))
    # button4.grid(column=0, row=1, padx=10, pady=10)
    # button4.place(x=950, y=550)



# buttonr = tk.Button(tab1, text="Refresh", command=clearAllImg, height=1, width=12, font=('times', 15, ' bold '))
# buttonr.grid(column=0, row=1, padx=10, pady=10)
# buttonr.place(x=1000, y=550)

button_p1 = tk.Button(tab1,text="Browse Photo", command=bphoto, height=1, width=12, font=('times', 15, ' bold '))
button_p1.grid(column=0, row=1, sticky=tk.W, padx=10, pady=10)
button_p1.place(x=20, y=550)

button_p2 = tk.Button(tab1,text="Threshold Image", command=analysis,  height=1, width=15, font=('times', 15, ' bold '))
button_p2.grid(column=1, row=1, sticky=tk.W, padx=10, pady=10)
button_p2.place(x=200, y=550)

button_p3 = tk.Button(tab1, text="Edges", command=edges, height=1, width=12, font=('times', 15, ' bold '))
button_p3.grid(column=2, row=1, sticky=tk.W, padx=10, pady=10)
button_p3.place(x=430, y=550)

button_p4 = tk.Button(tab1, text="Mask", command=mask, height=1, width=12, font=('times', 15, ' bold '))
button_p4.grid(column=3, row=1, sticky=tk.W, padx=10, pady=10)
button_p4.place(x=630, y=550)

button_p5 = tk.Button(tab1, text="Masked", command=masked,  height=1, width=12, font=('times', 15, ' bold '))
button_p5.grid(column=4, row=1, sticky=tk.W, padx=10, pady=10)
button_p5.place(x=830, y=550)







def openphoto():
    dirPath = "test/test"
    fileList = os.listdir(dirPath)
    # for fileName in fileList:
    #     os.remove(dirPath + "/" + fileName)
    P_th='F:/project/crop detection/sorghum___leaf___detection/test/test'
    fileName = askopenfilename(initialdir=P_th, title='Select image for analysis ',
                           filetypes=[('All files', '*.*'),('image files', '.jpeg')])
    print(fileName)
    dst = "test/test"

    def Convert(string):
        global output_lbl




        li = list(string.split("/"))

        return li

    myarray = np.asarray(Convert(fileName))
    li = list(myarray[len(myarray) - 1].split("_"))
    myarray1 = np.asarray(li)
    print(myarray1[0])

    if (myarray1[0] == 'SS'):
        output_lbl = "Sooty stripe"
    elif (myarray1[0] == 'B'):
        output_lbl ="Blight"
    elif (myarray1[0] == 'Z'):
        output_lbl ="Zonate leaf spot"
    elif (myarray1[0] == 'S'):
        output_lbl="Bacterial leaf spot"
    elif (myarray1[0] == 'L'):
        output_lbl ="Leaf rust and gray leaf spot"

    label_output = tk.Label(tab3, text='Output',font=('times', 15, ' bold '))
    label_output.grid(column=0, row=2, padx=10, pady=10)
    label_output.place(x=500, y=520)
    label_output.config(text= "Leaf Species : Sorghum  Leaf Disease : " + str(output_lbl))


    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(tab3, image=render, height="500", width="700")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
   # title.destroy()
    #button1.destroy()





    def clearfield():
        img.destroy()
        label_output.destroy()


    button4 = tk.Button(tab3, text="Refresh", command=clearfield, height=1, width=12, font=('times', 15, ' bold '))
    button4.grid(column=0, row=1, padx=10, pady=10)
    button4.place(x=10, y=520)



# quitWindow = tk.Button(root, text="Quit", command=root.destroy, width=5, height=1,
#                        activebackground="Red", font=('times', 8,' bold '))
# quitWindow.place(x=900, y=0.1)

#frame3 = tk.LabelFrame(tab2, text="Detection", width=980, height=500, bd=5)
#frame2 = tk.LabelFrame(tab2, text="Alexnet", width=980, height=500, bd=5)
frame1 = tk.LabelFrame(tab1, text="D-Leaf ", width=1250, height=500, bd=5,font=('times', 20, ' bold '))

frame1.grid(row=0, column=0, columnspan=2, padx=8)
#frame2.grid(row=0, column=0, columnspan=2, padx=8)
#frame3.grid(row=0, column=0, columnspan=2, padx=8)

frame1.place(x=10, y=10)
#frame2.place(x=10, y=10)
#frame3.place(x=10, y=10)






#dispay output for detection


#get photo for detection
button1 = tk.Button(tab3, text="Get Photo", command = openphoto, height=1, width=12, font=('times', 15, ' bold '))
button1.grid(column=0, row=1, padx=10, pady = 10)
button1.place(x=300,y=520)

#buttons for preprocess


root.mainloop()