import cv2 
import numpy as np
import os
import torch
label_dir = r'C:\Users\Asus\Desktop\AI\SegmentData\FullData\train\label'
label_mapping = {0: 0, 255: 1}

def convert_label(label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in label_mapping.items():
                label[temp == k] = v
        return label
# zero = 0.0
# none_zero = 0.0

# for file in os.listdir(label_dir):
#     label = cv2.imread(os.path.join(label_dir, file), cv2.IMREAD_GRAYSCALE)
#     label = label[90:170,:]
#     label = (np.where(label > 255/2, 255, 0).astype('uint8'))
#     label = convert_label(label)
#     none_zero = none_zero + np.count_nonzero(label)
#     zero = zero + 25600 - np.count_nonzero(label)
    
# print("zero: ", zero)
# print("none_zero: ", none_zero)

CLASS_WEIGHT = torch.FloatTensor([81636981.0 , 117915019.0])

class_freg = torch.FloatTensor(CLASS_WEIGHT)
weight = 1/torch.log1p(class_freg)
weight = 2 * weight / torch.sum(weight)

print(weight)

