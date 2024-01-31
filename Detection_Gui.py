import tkinter as tk
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import sys
import PIL
from PIL import Image, ImageTk


window = tk.Tk()

window.title("leaf")

window.geometry("500x510")
window.configure(background ="lightgreen")

title = tk.Label(text="Click below to choose picture for testing disease....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
def bact():
    window.destroy()
    window1 = tk.Tk()

    window1.title("leaf")

    window1.geometry("500x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "plant "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " plants"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def vir():
    window.destroy()
    window1 = tk.Tk()

    window1.title("leaf")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = " "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " plant"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def latebl():
    window.destroy()
    window1 = tk.Tk()

    window1.title("leaf")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = " "
    remedies = tk.Label(text=rem, background="light",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " plant"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def analysis():
    import warnings
    warnings.filterwarnings('ignore')  # suppress import warnings

    import os
    import cv2
    import tflearn
    import numpy as np
    import tensorflow as tf
    from random import shuffle
    from tqdm import tqdm
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    ''' <global actions> '''

    TRAIN_DIR = 'train/train'
    TEST_DIR = 'test/test'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'sorghumleafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
    tf.logging.set_verbosity(tf.logging.ERROR)  # suppress keep_dims warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow gpu logs
    tf.reset_default_graph()

    ''' </global actions> '''



    def label_leaves(leaf):

        leaftype = leaf[0]
        ans = [0, 0, 0, 0]

        if leaftype == 'h':
            ans = [1, 0, 0, 0]
        elif leaftype == 'b':
            ans = [0, 1, 0, 0]
        elif leaftype == 'v':
            ans = [0, 0, 1, 0]
        elif leaftype == 'l':
            ans = [0, 0, 0, 1]

        return ans

    def create_training_data():

        training_data = []

        for img in tqdm(os.listdir(TRAIN_DIR)):
            label = label_leaves(img)
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])

        shuffle(training_data)
        np.save('train_data.npy', training_data)

        return training_data

def main():

        train_data = create_training_data()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model Loaded')

        train = train_data[:-500]
        test = train_data[-500:]

        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)

import warnings
warnings.filterwarnings('ignore')  # suppress import warnings


def detect():
    import os
    import sys
    import cv2
    import tflearn
    import numpy as np
    import tensorflow as tf
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'sorghumleafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
    tf.logging.set_verbosity(tf.logging.ERROR)  # suppress keep_dims warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow gpu logs

    def process_verify_data(filepath):

        verifying_data = []

        img_name = filepath.split('.')[0]
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        verifying_data = [np.array(img), img_name]

        np.save('verify_data.npy', verifying_data)

        return verifying_data

    def analysis(filepath):

        verify_data = process_verify_data(filepath)

        str_label = "Cannot make a prediction."
        status = "Error"

        tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model loaded successfully.')
        else:
            print('Error: Create a model using neural_network.py first.')

        img_data, img_name = verify_data[0], verify_data[1]

        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'Sooty stripe'
        elif np.argmax(model_out) == 1:
            str_label = 'Bacterial leaf spot'
        elif np.argmax(model_out) == 2:
            str_label = 'Zonate leaf spot'
        elif np.argmax(model_out) == 3:
            str_label = 'Leaf rust and gray leaf spot'
        elif np.argmax(model_out) == 4:
            str_label = 'Sorghum leaf blight'

        if str_label == 'Healthy':
            status = 'Healthy'
        else:
            status = 'Unhealthy'

        result = 'Status: ' + status + '.'

        if (str_label != 'Healthy'): result += '\nDisease: ' + str_label + '.'

        return result

    def main():
        filepath = input("Enter Image File Name:\n")
        print(analysis(filepath))

    if __name__ == '__main__': main()


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
        output_lbl ="Sorghum leaf blight"
    elif (myarray1[0] == 'Z'):
        output_lbl ="Zonate leaf spot"
    elif (myarray1[0] == 'S'):
        output_lbl="Bacterial leaf spot"
    elif (myarray1[0] == 'L'):
        output_lbl ="Leaf rust and gray leaf spot"

    label_output.config(text= "Leaf Species....." + str(output_lbl))


    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()

label_output=tk.Label(text='Output')   #, height="20", width="10")
label_output.grid(column=0, row=2,padx=10, pady = 10)

button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()



