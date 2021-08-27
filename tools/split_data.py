
import os
import shutil
import numpy as np 
from sklearn.model_selection import train_test_split

train_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\train'
test_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\test'
valid_dir = r'C:\Users\Asus\Desktop\AI\DataUIT\valid'


train_dir1 = r'C:\Users\Asus\Desktop\AI\DataUIT_1\train'
test_dir1 = r'C:\Users\Asus\Desktop\AI\DataUIT_1\test'
valid_dir1 = r'C:\Users\Asus\Desktop\AI\DataUIT_1\valid'


train_files = []
for file in os.listdir(train_dir):
    train_files.append(file)

test_files = []
for file in os.listdir(test_dir):
    test_files.append(file)

valid_files = []
for file in os.listdir(valid_dir):
    valid_files.append(file)


train_files = np.array(train_files)
test_files = np.array(test_files)
valid_files = np.array(valid_files)

print(train_files.shape, test_files.shape, valid_files.shape)

train, new_train = train_test_split(train_files, test_size=0.5, random_state=42)
test, new_test = train_test_split(test_files, test_size=0.5, random_state=42)
valid, new_valid = train_test_split(valid_files, test_size=0.5, random_state=42)

for file in new_train:
    current_path = os.path.join(train_dir, file)
    destination_path = os.path.join(train_dir1, file)
    shutil.move(current_path, destination_path)

for file in new_test:
    current_path = os.path.join(test_dir, file)
    destination_path = os.path.join(test_dir1, file)
    shutil.move(current_path, destination_path)

for file in new_valid:
    current_path = os.path.join(valid_dir, file)
    destination_path = os.path.join(valid_dir1, file)
    shutil.move(current_path, destination_path)



