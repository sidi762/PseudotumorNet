'''
Dataset for brain tumor classification, T1+T2 hybrid
Sidi Liang, 2022-2023
'''

import math
import os
import pathlib
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from typing import Tuple, List, Dict

class CustomTumorDataset(Dataset):

    def __init__(self, root_dir, sets):
        self.phase = sets.phase
        if self.phase == "test":
            with open(sets.data_list, 'r') as f:
                self.data_list = [line.strip() for line in f]
                print("Processing {} pairs of data for classification".format(len(self.data_list)))
                #print(self.data_list)
        elif self.phase == "train":
            self.classes, self.class_to_idx = self.__find_classes__(root_dir)

        self.paths = list(pathlib.Path(root_dir).glob("*/*"))#folder containing the T1 and T2 images for one patient
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.with_augmentation = sets.with_augmentation
        self.t1_image_list = []
        self.t2_image_list = []

    def __nii2tensorarray__(self, data):
        # [z, y, x] = data.shape
        # new_data = np.reshape(data, [z, y, x])
        # new_data = new_data.astype("float32")

        # return new_data
        return data.astype("float32")

    # Make function to find classes in target directory
    def __find_classes__(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folder names in a target directory.

        Assumes target directory is in standard image classification format.

        Args:
            directory (str): target directory to load classnames from.

        Returns:
            Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

        Example:
            find_classes("food_images/train")
            >>> (["class_1", "class_2"], {"class_1": 0, ...})
        """
        # 1. Get the class names by scanning the target directory
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        # 2. Raise an error if class names not found
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

        # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        if self.phase == "test":
            return len(self.data_list)
        else:
            return len(self.paths)

    def __getitem__(self, idx):
        if self.phase == "train":
            # read image and labels
            patient_path = self.paths[idx]
            
            if self.with_augmentation is False:
                t1_image_path = list(patient_path.glob("*t1.nii.gz"))
                # print(patient_path)
                t2_image_path = list(patient_path.glob("*t2.nii.gz"))
                t1_image_path = t1_image_path[0]
                assert os.path.isfile(t1_image_path)
                self.t1_image_list.append(t1_image_path)
                t2_image_path = t2_image_path[0]
                self.t2_image_list.append(t2_image_path)
                assert os.path.isfile(t2_image_path)
            else:
                t1_image_path = list(patient_path.glob("*t1_aug.nii.gz"))
                # print(patient_path)
                t2_image_path = list(patient_path.glob("*t2_aug.nii.gz"))
                if len(t1_image_path) < 1:
                    print("Error: " + str(patient_path))
                t1_image_path = t1_image_path[0]
                assert os.path.isfile(t1_image_path)
                self.t1_image_list.append(t1_image_path)
                t2_image_path = t2_image_path[0]
                self.t2_image_list.append(t2_image_path)
                assert os.path.isfile(t2_image_path)
            
            class_name  = self.paths[idx].parent.name
            t1_img = nibabel.load(t1_image_path)
            t1_img_array = t1_img.get_fdata()
            t2_img = nibabel.load(t2_image_path)
            t2_img_array = t2_img.get_fdata()
            assert t1_img_array is not None
            assert t2_img_array is not None
            img_array = np.array([t1_img_array, t2_img_array])
            # data processing
            #t1_img_array, t2_img_array = self.__training_data_process__(t1_img, t2_img)
            # affine = t1_img.affine
            # t1_after_processing = nibabel.Nifti1Image(t1_img_array, affine)
            # nibabel.save(t1_after_processing, 't1_after_processing.nii.gz')  
            
            img_array = self.__training_data_process__(img_array)

            # 2 tensor array
            # t1_img_array = self.__nii2tensorarray__(t1_img_array)
            # t2_img_array = self.__nii2tensorarray__(t2_img_array)
            # img_array = np.array([t1_img_array, t2_img_array])
            img_array = self.__nii2tensorarray__(img_array)
            #print(img_array.shape)
            class_idx = self.class_to_idx[class_name]

            return img_array, class_idx, patient_path.__str__()
        elif self.phase == "test":
            # print(idx)
            patient_data = self.data_list[idx].split()
            if len(patient_data) != 2:
                raise ValueError("Each line in the data list should contain two MRI image paths.")

            # Load the two MRI images
            t1_path, t2_path = patient_data
            # t1_img_name = os.path.join(self.root_dir, t1_ith_info[0])
            # t2_img_name = os.path.join(self.root_dir, t1_ith_info[0])

            # Get the parent directory of the images
            patient_path = os.path.dirname(t1_path)
            self.t1_image_list.append(t1_path)
            self.t2_image_list.append(t2_path)

            assert os.path.isfile(t1_path)
            assert os.path.isfile(t2_path)
            # print("t1: ", t1_path, "t2: ", t2_path)
            t1_img = nibabel.load(t1_path)
            t2_img = nibabel.load(t2_path)
            t1_img_array = t1_img.get_fdata()
            t2_img_array = t2_img.get_fdata()
            assert t1_img is not None
            assert t2_img is not None

            img_array = np.array([t1_img_array, t2_img_array])
            # data processing
            # t1_img_array, t2_img_array = self.__testing_data_process__(t1_img, t2_img)
            img_array = self.__testing_data_process__(img_array)
        
            # 2 tensor array
            # t1_img_array = self.__nii2tensorarray__(t1_img_array)
            # t2_img_array = self.__nii2tensorarray__(t2_img_array)
            img_array = self.__nii2tensorarray__(img_array)

            return img_array, patient_path

    def __drop_invalid_range_fixed__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        [img_d, img_h, img_w] = volume.shape

        [max_z, max_h, max_w] = [img_d, img_h, img_w]
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __stack_drop_invalid_range__(self, volume, label=None):
        first_vol = volume[0]
        sec_vol = volume[1]
        first_vol, sec_vol = self.__drop_invalid_range__(first_vol, sec_vol)
        return np.array([first_vol, sec_vol])

    def __random_flip__(self, data, data2):
        # Randomly choose whether to flip or not
        flip = np.random.choice([True, False])
        # flip = 1
        
        # Flip the data along the selected axis if flip is True
        if flip:
            data = np.flip(data, axis=0)
            data2 = np.flip(data2, axis=0)
        
        return data, data2
    def __random_center_crop__(self, data, label=None):
        from random import random
        """
        Random crop
        """
        if label is not None:
            target_indexs = np.where(label>0)
        else:
            target_indexs = np.where(data>0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        if label is None:
            return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
        else:
            return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzero region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        # out = np.clip(out, -5, 5) # Clip the values to the range [-5, 5]
        # out = (out + 5) / 10 # Rescale the values to the range [0, 1]
        # out_random = np.random.normal(0, 1, size = volume.shape)
        # out[volume == 0] = out_random[volume == 0] # Add Gausian Noise to the zeros
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [width, depth, height] = data.shape
        scale = [self.input_W*1.0/width, self.input_D*1.0/depth, self.input_H*1.0/height]
        data = ndimage.zoom(data, scale, order=0)

        return data

    def __stack_resize_data__(self, data):
        """
        Resize the data to the input size
        """
        first_vol = data[0]
        sec_vol = data[1]
        first_vol = self.__resize_data__(first_vol)
        sec_vol = self.__resize_data__(sec_vol)
        return np.array([first_vol, sec_vol])
    
    def __crop_data__(self, data, label=None):
        """
        Random crop with different methods:
        """
        if label is None:
            # random center crop
            data = self.__random_center_crop__ (data)
            return data
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        return data, label

    def __training_data_process__(self, data, data2=None, label=None):
        if label is None:
            # For classification
            if data2: # Process two volumes (T1 and T2 in our case)
                # data = data.get_fdata()
                # data2 = data2.get_fdata()  # get data from nii and returns an array.
                data, data2 = self.__drop_invalid_range__(data, data2) # drop out the invalid range
                # data, data2 = self.__crop_data__(data, data2) # crop data
                # resize data
                data = self.__resize_data__(data) 
                data2 = self.__resize_data__(data2)
                # random flip
                # data, data2 = self.__random_flip__(data, data2)
                # normalization
                data = self.__itensity_normalize_one_volume__(data)
                data2 = self.__itensity_normalize_one_volume__(data2)
                return data, data2
            else: # For single volume processing
                # data = data.get_fdata()
                #data = data[:,:,:,0]
                # drop out the invalid range
                # data = self.__drop_invalid_range__(data)
                data = self.__stack_drop_invalid_range__(data)
                # crop data
                # data = self.__crop_data__(data)
                # resize data
                # data = self.__resize_data__(data)
                data = self.__stack_resize_data__(data)
                # normalization data
                data = self.__itensity_normalize_one_volume__(data)
                return data
        else:
            # crop data according net input size
            # For segmentation, WIP
            # data = data.get_fdata()
            # label = label.get_fdata()

            # drop out the invalid range
            data, label = self.__drop_invalid_range__(data, label)

            # crop data
            #data, label = self.__crop_data__(data, label)

            # resize data
            # data = self.__resize_data__(data)
            # label = self.__resize_data__(label)

            # normalization datas
            data = self.__itensity_normalize_one_volume__(data)

            return data, label

    def __testing_data_process__(self, data, data2=None, crop=False, segmentation=False):
        if segmentation is False:
            # For classification
            if data2: 
                # Process two volumes (T1 and T2 in our case)
                # data = data.get_fdata()
                # data2 = data2.get_fdata()  # get data from nii and returns an array.
                # data, data2 = self.__drop_invalid_range__(data, data2) # drop out the invalid range
                # if crop:
                #     data, data2 = self.__crop_data__(data, data2) # crop data
                # resize data
                data = self.__resize_data__(data) 
                data2 = self.__resize_data__(data2)
                # normalization
                data = self.__itensity_normalize_one_volume__(data)
                data2 = self.__itensity_normalize_one_volume__(data2)
                return data, data2
            else: # For single volume processing
                # get data from nii and returns an array. 
                # data = data.get_fdata()
                #data = data[:,:,:,0]
                # drop out the invalid range
                # data = self.__drop_invalid_range__(data)
                data = self.__stack_drop_invalid_range__(data)
                # if crop:
                #     data = self.__crop_data__(data) # crop data
                # resize data
                # data = self.__resize_data__(data)
                data = self.__stack_resize_data__(data)
                # normalization
                data = self.__itensity_normalize_one_volume__(data)
                return data
        else:
            # crop data according net input size
            # For segmentation, WIP
            data = data.get_data()
            label = label.get_data()

            # drop out the invalid range
            # data, label = self.__drop_invalid_range__(data, label)

            # crop data
            #data, label = self.__crop_data__(data, label)

            # resize data
            data = self.__resize_data__(data)
            label = self.__resize_data__(label)

            # normalization datas
            data = self.__itensity_normalize_one_volume__(data)

            return data, label
