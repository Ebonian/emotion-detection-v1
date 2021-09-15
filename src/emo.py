from logging import root
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "    Angry   ", 1: "    Disgusted    ", 2: "    Fearful    ",
                3: "    Happy   ", 4: "     Neutral    ", 5: "    Sad    ", 6: "    Surprised    "}

cur_path = os.path.dirname(os.path.abspath(__file__))

emo_dist = {0: cur_path+"./emo/angry.png", 1: cur_path+"./emo/disgusted.png", 2: cur_path+"./emo/fearful.png", 3: cur_path +
            "./emo/happy.png", 4: cur_path+"./emo/neutral.png", 5: cur_path+"./emo/sad.png", 7: cur_path+"./emo/surpirsed.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]
global frame_number


def show_subject():
    # use camera
    cap1 = cv2.VideoCapture('./data/examples/video.mp4')
    # test camera
    if not cap1.isOpened():
        print("Can't open the video source")
    # global var
    global frame_number
    # get length of frame
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    # check frame
    frame_number += 1
    if frame_number >= length:
        exit()
    # set next frame
    cap1.set(1, frame_number)
    # read the frame
    flag1, frame1 = cap1.read()

    frame1 = cv2.resize(frame1, (600, 500))
    # load pre-trained data
    bounding_box = cv2.CascadeClassifier(
        './data/haarcascades/haarcascade_frontalface_default.xml')
    # covert to b&w
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # detect face
    num_faces = bounding_box.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        # create rectangle
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # cropped & resize
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # predict emotion from emotion model
        prediction = emotion_model.predict(cropped_img)
        # get index
        maxindex = int(np.argmax(prediction))
        # from dict
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print("error")
    # update main window
    elif flag1:
        # get frame
        global last_frame1
        last_frame1 = frame1.copy()
        # covert to rgb
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        # genereate from pil
        img = Image.fromarray(pic)
        # tklabel
        imgtk = ImageTk.PhotoImage(image=img)
        # set imagetk value from label
        lmain.imagetk = imgtk

        lmain.configure(image=imgtk)
        # update root
        root.update()

        lmain.after(10, show_subject)
    # q for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_avatar():
    # read emo
    frame2 = cv2.imread(emo_dist[show_text[0]])
    # convert to rgb
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    # configure text & font
    lmain3.configure(
        text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)


if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#cdcdcd", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Emote Recognition")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitButton = Button(root, text='Quit', fg="red", command=root.destroy, font=(
        'arial', 25, 'bold')).pack(side=BOTTOM)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()
