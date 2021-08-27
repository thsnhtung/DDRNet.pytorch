import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

dir = r'C:\Users\Asus\Downloads\Data\Data_11k5_pkl'
train_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\train'
test_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\test'
valid_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\valid'

files = []
for file in os.listdir(dir):
    files.append(file)

files = np.array(files)
print(files.shape)
train, test = train_test_split(files, test_size=0.2, random_state=42)

test, valid = train_test_split(test, test_size=0.5, random_state=42)

for file in train:
    current_path = os.path.join(dir, file)
    destination_path = os.path.join(train_dir, file)
    shutil.move(current_path, destination_path)

for file in test:
    current_path = os.path.join(dir, file)
    destination_path = os.path.join(test_dir, file)
    shutil.move(current_path, destination_path)

for file in valid:
    current_path = os.path.join(dir, file)
    destination_path = os.path.join(valid_dir, file)    
    shutil.move(current_path, destination_path)