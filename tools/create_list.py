import os 


root = r'C:\Users\Asus\Desktop\AI\DataUIT'

list_dir = r'C:\Users\Asus\Desktop\DDRNet.pytorch\Datas\list\uit'

train_dir = os.path.join(root, 'valid')

img_dir = os.path.join(train_dir, 'image')
label_dir = os.path.join(train_dir, 'label')

out_file = open(os.path.join(list_dir, "valid.txt"), "w") 

for file in os.listdir(img_dir):
    data_line = "valid/image/" + file + "\t"  "valid/label/" + file + "\n"
    out_file.write(data_line) 

out_file.close()

