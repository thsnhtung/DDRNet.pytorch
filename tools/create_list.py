import os 


root = r'C:\Users\Asus\Desktop\AI\SegmentData\FullData'

list_dir = r'C:\Users\Asus\Desktop\DDRNet.pytorch'

train_dir = os.path.join(root, 'train')

img_dir = os.path.join(train_dir, 'image')
label_dir = os.path.join(train_dir, 'label')

out_file = open(os.path.join(list_dir, "train.txt"), "w") 

for file in os.listdir(img_dir):
    data_line = "image/" + file + "\t" + "label/" + file + "\n"
    out_file.write(data_line) 

out_file.close()

