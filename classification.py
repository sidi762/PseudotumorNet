import torch.nn.functional as F
import torch
from tqdm import  tqdm
import nibabel as nib
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
sys.path.append('./MedicalNet/dual')
import MedicalNet.dual.model as MMModel
from settings import class_parse_opts
from brain.pseudotumor_classi_net.MedicalNet.dual.datasets.wch_dataset import CustomTumorDataset

from medcam import medcam

#Classification
def classification(data_loader, model, sets):
    predicted_labels = []
    class_probs = []  # To store class probabilities
    results = []
    model.eval() # for testing
    with torch.no_grad():
        for batch_id, (volumes, patient_path) in tqdm(enumerate(data_loader), \
                                        total=len(data_loader), \
                                        postfix='seg', \
                                        ncols=100, ):
            # forward
            if not sets.no_cuda:
                volumes = volumes.cuda()

            out_class = model(volumes)
            probs = F.softmax(out_class, dim=1)
            class_probs.extend(probs.cpu().numpy())  # Store class probabilities

            _, predicted = torch.max(out_class.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())  # Store predicted labels
            results.append([patient_path, predicted.cpu(), probs.cpu().numpy()])

    return [results, predicted_labels, class_probs]




if __name__ == '__main__':
    gpu_id = [0]
    classi_sets = class_parse_opts() # Configuration for classification model
    torch.manual_seed(classi_sets.manual_seed)
    classi_sets.model = 'resnet'
    #classi_sets.resume_path = "MedicalNet/MedicalNet_dual/trails/models0415_2/resnet_18_dualseq_epoch_149_val_loss_0.5189778804779053_accuracy_72.0.pth.tar" #All returns 1
    #classi_sets.resume_path = "MedicalNet/MedicalNet_dual/trails/models0415_2/resnet_18_dualseq_epoch_70_val_loss_0.6459382176399231_accuracy_80.0.pth.tar"
    #classi_sets.resume_path = "resnet_18_dualseq_epoch_5_val_loss_0.5973857045173645_accuracy_76.0.pth.tar"
    #classi_sets.resume_path = "MedicalNet/MedicalNet_dual/trails/models0415/resnet_18_dualseq_epoch_54_val_loss_0.6263759136199951_accuracy_64.0.pth.tar"
    #classi_sets.resume_path = "resnet_18_dualseq_128_epoch_20_val_loss_0.7904279232025146_accuracy_76.0.pth.tar"
    classi_sets.resume_path = "MedicalNet/dual/resnet_18_dualseq_epoch_81_val_loss_0.006248984485864639_accuracy_100.0.pth.tar"
    classi_sets.data_list = "data/aug_list.txt"
    #classi_sets.data_list = "data/brain_seg_test_list_t1t2.txt"
    #classi_sets.data_list = "data/brain_seg_test_list.txt"
    #classi_sets.data_list = "MedicalNet/MedicalNet_dual/dataset_t1_t2/data_t1_t2_match_train.txt"
    classi_sets.data_root = "data/Preprocessed"
    classi_sets.model_depth = 18
    classi_sets.resnet_shortcut = 'A'
    classi_sets.batch_size = 1
    classi_sets.gpu_id = gpu_id
    classi_sets.input_W = 128
    classi_sets.input_H = 128
    classi_sets.input_D = 128
    classi_sets.phase = 'test'
    # Getting model
    checkpoint = torch.load(classi_sets.resume_path)
    model, _ = MMModel.generate_model(classi_sets)
    if classi_sets.no_cuda:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model = medcam.inject(model, backend='gcampp', layer='auto', output_dir="attention_maps", save_maps=True)
    testing_data =CustomTumorDataset(classi_sets.data_root, classi_sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    results = classification(data_loader, model, classi_sets)
    for result in results[0]:
        print(result, "\n")
