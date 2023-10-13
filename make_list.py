import os

root_path = '/users/Gaoyu/DL/liangsidi/pseudotumor_classification_net/MedicalNet/MedicalNet_dual/dataset_t1_t2'
train_path = 'data_t1_t2_match'
val_path = 'data_val_t1_t2_match'
train_path_full = os.path.join(root_path, train_path)
val_path_full = os.path.join(root_path, val_path)

train_list_path = os.path.join(root_path,train_path_full+'_train.txt')
val_list_path = os.path.join(root_path,val_path_full+'_valid.txt')

train_path_full_glioma = os.path.join(train_path_full, 'glioma')
train_path_full_pseudo = os.path.join(train_path_full, 'pseudotumor')

val_path_full_glioma = os.path.join(val_path_full, 'glioma')
val_path_full_pseudo = os.path.join(val_path_full, 'pseudotumor')


train_list = open(train_list_path, 'w')
valid_list = open(val_list_path, 'w')

# Iterate all files in train_path_full and add its full path to train_list
for root, dirs, files in os.walk(train_path_full_glioma):
    for dir in dirs:
        imgs = ["", ""]
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("t1.nii.gz"):
                imgs[0] = os.path.join(train_path_full_glioma, dir, file)
            if file.endswith("t2.nii.gz"):
                imgs[1] = os.path.join(train_path_full_glioma, dir, file)
        train_list.write(imgs[0] + ' ' + imgs[1] + '\n')

for root, dirs, files in os.walk(train_path_full_pseudo):
    for dir in dirs:
        imgs = ["", ""]
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("t1.nii.gz"):
                imgs[0] = os.path.join(train_path_full_pseudo, dir, file)
            if file.endswith("t2.nii.gz"):
                imgs[1] = os.path.join(train_path_full_pseudo, dir, file)
        train_list.write(imgs[0] + ' ' + imgs[1] + '\n')

# Iterate all files in val_path_full and add its full path to valid_list
for root, dirs, files in os.walk(val_path_full_glioma):
    for dir in dirs:
        imgs = ["", ""]
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("t1.nii.gz"):
                imgs[0] = os.path.join(val_path_full_glioma, dir, file)
            if file.endswith("t2.nii.gz"):
                imgs[1] = os.path.join(val_path_full_glioma, dir, file)
        valid_list.write(imgs[0] + ' ' + imgs[1] + '\n')

for root, dirs, files in os.walk(val_path_full_pseudo):
    for dir in dirs:
        imgs = ["", ""]
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("t1.nii.gz"):
                imgs[0] = os.path.join(val_path_full_pseudo, dir, file)
            if file.endswith("t2.nii.gz"):
                imgs[1] = os.path.join(val_path_full_pseudo, dir, file)
        valid_list.write(imgs[0] + ' ' + imgs[1] + '\n')


train_list.close()
valid_list.close()
