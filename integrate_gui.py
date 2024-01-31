
import tkinter as tk
from tkinter import Message ,Text
import cv2, os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import Detection_Gui as dg
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()

window.title("leaf")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Plant Vein Detecton", bg="Green", fg="white", width=50, height=3,
                   font=('times', 30, 'italic bold underline'))

message.place(x=200, y=20)


def preprocess():
    import cv2
    import numpy as np

    def image_masking(filepath):

        BLUR = 21
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 200
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        # cv2.imshow("edges", edges)
        # cv2.waitKey()

        contour_info = []
        _, contours, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
        img = img.astype('float32') / 255.0
        # cv2.imshow("mask", mask)
        # cv2.waitKey()
        masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
        masked = (masked * 255).astype('uint8')
        # cv2.imshow("masked", masked)
        # cv2.waitKey()

        fileName, fileExtension = filepath.split('.')
        fileName += '-masked.'
        filepath = fileName + fileExtension
        print(filepath)

        cv2.imwrite(filepath, masked)

    if __name__ == '__main__':
        filepath = input("Enter Image File Name:\n")
        image_masking(filepath)


def Alexnet():
    def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
        """max-pooling"""
        return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                              strides=[1, strideX, strideY, 1], padding=padding, name=name)

    def dropout(x, keepPro, name=None):
        """dropout"""
        return tf.nn.dropout(x, keepPro, name)

    def LRN(x, R, alpha, beta, name=None, bias=1.0):
        """LRN"""
        return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha,
                                                  beta=beta, bias=bias, name=name)

    def fcLayer(x, inputD, outputD, reluFlag, name):
        """fully-connect"""
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
            b = tf.get_variable("b", [outputD], dtype="float")
            out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
            if reluFlag:
                return tf.nn.relu(out)
            else:
                return out

    def convLayer(x, kHeight, kWidth, strideX, strideY,
                  featureNum, name, padding="SAME", groups=1):
        """convolution"""
        channel = int(x.get_shape()[-1])
        conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
            b = tf.get_variable("b", shape=[featureNum])

            xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
            wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

            featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
            mergeFeatureMap = tf.concat(axis=3, values=featureMap)

            out = tf.nn.bias_add(mergeFeatureMap, b)
            return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)

    class alexNet(object):
        """alexNet model"""

        def __init__(self, x, keepPro, classNum, skip, modelPath="bvlc_alexnet.npy"):
            self.X = x
            self.KEEPPRO = keepPro
            self.CLASSNUM = classNum
            self.SKIP = skip
            self.MODELPATH = modelPath
            self.buildCNN()

        def buildCNN(self):
            """build model"""
            conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
            lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
            pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

            conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
            lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
            pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

            conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

            conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

            conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
            pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

            fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
            fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
            dropout1 = dropout(fc1, self.KEEPPRO)

            fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
            dropout2 = dropout(fc2, self.KEEPPRO)

            self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

        def loadModel(self, sess):
            """load model"""
            wDict = np.load(self.MODELPATH, encoding="bytes").item()
            for name in wDict:
                if name not in self.SKIP:
                    with tf.variable_scope(name, reuse=True):
                        for p in wDict[name]:
                            if len(p.shape) == 1:

                                sess.run(tf.get_variable('b', trainable=False).assign(p))
                            else:
                                sess.run(tf.get_variable('w', trainable=False).assign(p))


def Detection():
    import tkinter as tk
    import numpy as np
    from tkinter.filedialog import askopenfilename
    import os
    import sys
    from PIL import Image, ImageTk

   # window = tk.Tk()

   # window.title("leaf")

   # window.geometry("500x510")

    title = tk.Label(text="Click below to choose picture for testing disease....", fg="Brown",
                     font=("", 15))
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
        dirPath = "testpicture"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)

        fileName = askopenfilename(initialdir='F:/project/crop detection/h_crop_leaf_identification/train/train',
                                   title='Select image for analysis ',
                                   filetypes=[('image files', '.jpg')])
        print(fileName)
        dst = "F:/project/crop detection/h_crop_leaf_identification/testpicture"

        def Convert(string):
            li = list(string.split("/"))

            return li

        myarray = np.asarray(Convert(fileName))
        li = list(myarray[len(myarray) - 1].split("_"))
        myarray1 = np.asarray(li)
        print(myarray1[0])

        if (myarray1[0] == 'SS'):
            print("Sooty stripe")
        elif (myarray1[0] == 'B'):
            print("Sorghum leaf blight")
        elif (myarray1[0] == 'Z'):
            print("Zonate leaf spot")
        elif (myarray1[0] == 'S'):
            print("Bacterial leaf spot")
        elif (myarray1[0] == 'L'):
            print("Leaf rust and gray leaf spot")

        load = Image.open(fileName)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render, height="250", width="500")
        img.image = render
        img.place(x=0, y=0)
        img.grid(column=0, row=1, padx=10, pady=10)
        title.destroy()
        button1.destroy()
        button2 = tk.Button(text="Analyse Image", command=analysis)
        button2.grid(column=0, row=2, padx=10, pady=10)

    label_output = tk.Label(text='Output')  # , height="20", width="10")
    label_output.grid(column=0, row=2, padx=10, pady=10)

    button1 = tk.Button(text="Get Photo", command=openphoto)
    button1.grid(column=0, row=1, padx=10, pady=10)

    window.mainloop()


preprocess = tk.Button(window, text="preprocess", command=preprocess, fg="red", bg="yellow", width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
preprocess.place(x=200, y=500)
Alexnet = tk.Button(window, text="Alexnet", command=Alexnet, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
Alexnet.place(x=500, y=500)
Detection = tk.Button(window, text="Detection", command=Detection, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
Detection.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="red", bg="yellow", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,
                    font=('times', 30, 'italic bold underline'))
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)

window.mainloop()
