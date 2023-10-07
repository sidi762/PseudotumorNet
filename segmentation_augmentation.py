"""
    segmentation_augmentation.py
    Sidi Liang, 2023

    Take pre-processed brain MRI data (in the form of .nii.gz files)
    as input, and output the segmentation and augmentation result.
   
    1. Segmentation using 3D UNet
    2. Augmentation by overlaying the segmentation
        mask on the original image
    3. Generate the file list for subsequent classification using 3D ResNet


"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import MedicalNet.MedicalNet_dual.model as MMModel

#All configurations for segmentation and classification
from settings import class_parse_opts, seg_parse_opts

from tqdm import  tqdm
import nibabel as nib
import numpy as np
from scipy import ndimage

import sys
sys.path.append('./3D-UNet-seg-test')
from models import unet3d
import nii_augmentation as nii_aug

# Data Preprocessing for Segmentation
class seg_dataset(Dataset):
    def __init__(self, sets):
        with open(sets.data_list, 'r') as f:
            self.data_list = [line.strip() for line in f]
        print("Processing {} datas for seg".format(len(self.data_list)))
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.n_classes = sets.n_classes
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        # load img
        # info = self.data_list[idx].split(' ')
        # img_list = os.listdir(info[0])

        # for img in img_list:
        #     if 'T1' in img:
        #         t1_path = os.path.join(info[0], img)
        #     elif 'T2' in img:
        #         t2_path = os.path.join(info[0], img)
        
        # Get the pair of MRI image paths for the current patient
        patient_data = self.data_list[idx].split()

        if len(patient_data) != 2:
            raise ValueError("Each line in the data list should contain two MRI image paths.")
        
        # Load the two MRI images
        t1_path, t2_path = patient_data
        
        parent_directory = os.path.dirname(t1_path)
        
        if not os.path.isfile(t1_path):
            print("t1 path ", t1_path, " is not a file!")
        if not os.path.isfile(t2_path):
            print("t2 path ", t2_path, " is not a file!")

        assert os.path.isfile(t1_path)
        assert os.path.isfile(t2_path)

        # [D, H, W]
        t1_img = nib.load(t1_path)
        t1 = t1_img.get_fdata()
        t2_img = nib.load(t2_path)
        t2 = t2_img.get_fdata()
        # print(t1.shape)
        # print(t2.shape)
        # preprocesses
        img = np.array([t1, t2])
        affine = t1_img.affine
        img_array, pad1, pad2 = self.__drop_invalid_range__(img)
        [_, D, H, W] = img_array.shape
        img_array = self.__resize_data__(img_array)
        img_array = self.__itensity_normalize_one_vol_sub__(img_array)

        return img_array.astype("float32"), parent_directory, affine, [[D, H, W], pad1, pad2], [t1.astype("float32"), t2.astype("float32")]

    def __drop_invalid_range__(self, vol_sub):
        zero_value = vol_sub[0, 0, 0, 0]
        non_zeros_idx = np.where(vol_sub != zero_value)
        [_, max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [_, min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        return vol_sub[:, min_z:max_z, min_h:max_h, min_w:max_w], [min_z, min_h, min_w],[max_z, max_h, max_w]

    def __resize_data__(self, data):
        [_, depth, height, width] = data.shape
        scale = [1, self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.zoom(data, scale, order=0)
        return data

    def __itensity_normalize_one_vol_sub__(self, volume):
        [ch, _, _, _] = volume.shape
        for index in range(0, ch):
            vol_tmp = volume[index, :, :, :]
            pixels = vol_tmp[vol_tmp > 0]
            mean = pixels.mean()
            std = pixels.std()
            out = (vol_tmp - mean) / std
            out_random = np.random.normal(0, 1, size=vol_tmp.shape)
            out[vol_tmp == 0] = out_random[vol_tmp == 0]
            volume[index, :, :, :] = out
        return volume


# Segmentation
def segmentation(sets, img, img_path, pad, model):
        [D, H ,W] = [pad[0][0].item(), pad[0][1].item(), pad[0][2].item()]
        [pad_1_D, pad_1_H, pad_1_W] = [pad[1][0].item(), pad[1][1].item(), pad[1][2].item()]
        [pad_2_D, pad_2_H, pad_2_W] = [pad[2][0].item(), pad[2][1].item(), pad[2][2].item()]

        scale = [D/sets.input_D,H/sets.input_H,W/sets.input_W]
        img = img.cuda()
        output = model(img).detach().cpu().numpy()[0]
        # pixels = np.where((20 > output[1]) & (output[1]>output[2]))
        output = output.argmax(0).astype('uint8')
        # output[pixels] = 0
        output = ndimage.zoom(output, scale, order=0)
        output = np.pad(output, ((pad_1_D,256-pad_2_D),(pad_1_H, 256-pad_2_H),(pad_1_W,256-pad_2_W)), 'constant')

        # save_path = os.path.join(sets.output_path, img_name)
        save_path = img_path[0]
        if not os.path.exists(save_path): os.mkdir(save_path)
        #nib.save(nib.Nifti1Image(output, affine_out),
        #         os.path.join(save_path, img_name+'_seg.nii.gz'))
        return output


if __name__ == '__main__':
    sets = seg_parse_opts() # Configuration for segmentation model
    classi_sets = class_parse_opts() # Configuration for classification model

    # For segmentation:
    # loading ckpt
    checkpoint = torch.load(sets.checkpoint)
    # loading info
    sets.model = checkpoint['model']
    [sets.input_D, sets.input_H, sets.input_W] = checkpoint['img_shape']
    sets.n_channels = checkpoint['n_channels']
    sets.n_classes = checkpoint['n_classes']
    sets.data_list = "data/brain_seg_test_list.txt"
    gpu_id = [2]

    # For classification:
    torch.manual_seed(classi_sets.manual_seed)
    classi_sets.model = 'resnet'
    classi_sets.model_depth = 18
    classi_sets.resnet_shortcut = 'A'
    classi_sets.gpu_id = gpu_id
    classi_sets.input_W = 512
    classi_sets.input_H = 512
    classi_sets.input_D = 20
    classi_sets.data_list = "data/aug_list.txt"
    model, parameters = MMModel.generate_model(classi_sets) #3D Resnet 18


    # load segmentation model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
    model = unet3d.unet3d(sets.n_channels, sets.n_classes).cuda()
    model = torch.nn.DataParallel(model, device_ids=gpu_id)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # load data
    test_loader = DataLoader(seg_dataset(sets), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for batch_id, (img, img_path, affine, pad, original_imgs) in tqdm(enumerate(test_loader), \
                                                           total=len(test_loader), \
                                                           postfix='seg', \
                                                           ncols=100, ):
            # Segmentation
            output = segmentation(sets, img, img_path, pad, model)
            # Save segmentated image
            affine_out = affine.numpy()[0]
            save_path = img_path[0]
            img_name = img_path[0].split('/')[-1]
            nib.save(nib.Nifti1Image(output, affine_out),
                        os.path.join(save_path, img_name+'_seg.nii.gz'))

            # Augmentation
            aug_k = 0.5
            t1_orig = original_imgs[0]
            t2_orig = original_imgs[1]
            t1_orig = t1_orig.numpy()[0]
            t2_orig = t2_orig.numpy()[0]
            t1_augmented_output = nii_aug.augmentationWithFactorArray(imageArray=t1_orig, maskArray=output, k=aug_k)
            t2_augmented_output = nii_aug.augmentationWithFactorArray(imageArray=t2_orig, maskArray=output, k=aug_k)

            # Save augmentated image
            # print("img shape: ", t1_orig.shape)
            nib.save(nib.Nifti1Image(t1_augmented_output, affine_out),
                        os.path.join(save_path, img_name+'_t1_aug.nii.gz'))
            nib.save(nib.Nifti1Image(t2_augmented_output, affine_out),
                        os.path.join(save_path, img_name+'_t2_aug.nii.gz'))

            # Add the augmentated image file into a txt list for subsequent classification
            with open(os.path.join(save_path, '../aug_list.txt'), 'a') as f:
                f.write(save_path+"/"+img_name+'_t1_aug.nii.gz' + ' ' + save_path+"/"+img_name+'_t2_aug.nii.gz' + '\n')
