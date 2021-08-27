import pandas as pd
import numpy as np
import cv2

path = r"C:\Users\Asus\Downloads\Data\Data_11k5_pkl\image_VNU_551.pkl"

object = pd.read_pickle(path)

img = object[0]
label = object[1][:,:,3]


cv2.imshow("result", img)
cv2.imshow("label", label)

cv2.waitKey(0)