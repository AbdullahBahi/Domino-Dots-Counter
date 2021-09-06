import numpy as np
import cv2
from os import listdir, getcwd, remove
from os.path import join
from time import sleep

def get_frames():
    cwd = getcwd()
    vid_path = join(cwd,'video.mp4')
    cap = cv2.VideoCapture(vid_path)

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(join(cwd,'frames', 'f'+str(i)+'.jpg'), frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

    for j in range(675):
        if j < 10:
            remove(join(cwd,'frames', 'f'+str(j)+'.jpg'))

        elif (j-10)%75 == 0:
            img = cv2.imread(join(cwd,'frames', 'f'+str(j)+'.jpg'))
            cropped = img[266:582]
            resized = cv2.resize(cropped, (cropped.shape[1]+160,cropped.shape[0]+160), interpolation = cv2.INTER_AREA)
            cv2.imwrite(join(cwd,'frames', 'f'+str(j)+'.jpg'), resized)
            
        else:
            remove(join(cwd,'frames', 'f'+str(j)+'.jpg'))
    
    images = [f for f in listdir(join(cwd, 'frames'))]
    images.insert(1,images[-1])
    images = images[:-1]

    return images