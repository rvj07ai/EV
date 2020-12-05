import os, os.path, shutil,random ,numpy
# bug - to stop running this code
from sklearn.model_selection import train_test_split
images_path = "coco/images"
labels_path = "coco/labels"
list_files =  os.listdir(images_path)
image_ids = [img.split('.')[0] for img in list_files ]
# print([str(img.split('.')[0]+".txt")for img in list_files ])
image_index = list(range(0,len(image_ids)))
random.shuffle(image_index)
x_train ,x_valid = train_test_split(image_index,test_size=0.2,random_state =455)  

train_path = 'D:\\EagleView\\Object_Detection\\code\\yolo-5s-model\\data\\coco\\images\\train'
val_path = 'D:\\EagleView\\Object_Detection\\code\\yolo-5s-model\\data\\coco\\images\\val'
train_label_path = 'D:\\EagleView\\Object_Detection\\code\\yolo-5s-model\\data\\coco\\labels\\train'
val_label_path = 'D:\\EagleView\\Object_Detection\\code\\yolo-5s-model\\data\\coco\\labels\\val'
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(train_label_path):
    os.makedirs(train_label_path)
if not os.path.exists(val_label_path):
    os.makedirs(val_label_path)  

for x in x_train:
    old_image_path = os.path.join(images_path, image_ids[x] + ".jpg"  )
    new_image_path = os.path.join(train_path,  image_ids[x] + ".jpg" )
    shutil.copy(old_image_path, new_image_path)
    old_label_path = os.path.join(labels_path, image_ids[x] + ".txt" )
    new_label_path = os.path.join(train_label_path, image_ids[x] + ".txt" )
    shutil.copy(old_label_path, new_label_path)


for x_val in x_valid:
    old_image_path = os.path.join(images_path,  image_ids[x_val] + ".jpg" )
    new_image_path = os.path.join(val_path,  image_ids[x_val] + ".jpg" )
    shutil.copy(old_image_path, new_image_path)
    old_label_path = os.path.join(labels_path, image_ids[x_val] + ".txt" )
    new_label_path = os.path.join(val_label_path,   image_ids[x_val] + ".txt" )
    shutil.copy(old_label_path, new_label_path)
