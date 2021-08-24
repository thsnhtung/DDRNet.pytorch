import cv2 
import numpy as np
import random

img_path = r'C:\Users\Asus\Desktop\AI\SegmentData\Simulation\train\image\Roud1_1_31.png'


def brightness_augment(img, factor=0.5): 
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    for i in range(0,random.randint(1, 10)):
        shift_w_min = random.randint(0, 310)
        shift_w_max = random.randint(shift_w_min + 10 , 320)

        shift_h_min = random.randint(0, 70)
        shift_h_max = random.randint(shift_h_min+ 10 , 80)
        hsv[shift_h_min: shift_h_max,shift_w_min : shift_w_max, 2] = hsv[shift_h_min: shift_h_max,shift_w_min : shift_w_max, 2] * (factor + np.random.uniform() ) #scale channel V uniformly 
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
        rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

org_image = cv2.imread(img_path)
org_image = org_image[90:170,:]

# new_image = np.zeros(image.shape, image.dtype)
# alpha = 0.5 # Simple contrast control
# beta = -50    # Simple brightness control

# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for c in range(image.shape[2]):
#             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

while True:
    new_image = brightness_augment(org_image, factor= 0.5)
    cv2.imshow("origin", org_image)
    cv2.imshow('New Image', new_image)
    cv2.waitKey(800)
    cv2.destroyAllWindows()




