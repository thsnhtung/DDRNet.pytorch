import os
import shutil
import numpy as np 
import cv2
import pandas as pd

def ConvertPickle(dir, file):
    pickle_path = os.path.join(dir, file)
    object = pd.read_pickle(pickle_path)
    img = object[0]
    label = object[1][:,:,3] 


    img_path = os.path.join(dir, 'image', file[:-4] + ".png") 
    label_path = os.path.join(dir, 'label', file[:-4] + ".png") 
    cv2.imwrite(img_path, img)
    cv2.imwrite(label_path, label)
    os.remove(pickle_path)



dir = r'C:\Users\Asus\Desktop\AI\DataUIT_1\valid'
files = []
for file in os.listdir(dir):
    if file != 'label' and file != 'image': 
        ConvertPickle(dir, file)
        

