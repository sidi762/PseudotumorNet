import torch.nn.functional as F
import torch
from tqdm import  tqdm
import nibabel as nib
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
sys.path.append('./classification')
import classification.model as MMModel
from settings import class_parse_opts
from classification.datasets.wch_dataset import CustomTumorDataset

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
    gpu_id = [1]
    classi_sets = class_parse_opts() # Configuration for classification model
    torch.manual_seed(classi_sets.manual_seed)
    # classi_sets.model = 'resnet'
    classi_sets.model = 'convnext'
    #classi_sets.resume_path = "MedicalNet/dual/resnet_18_dualseq_epoch_81_val_loss_0.006248984485864639_accuracy_100.0.pth.tar"
    #classi_sets.resume_path = "resnet_18_dualseq_fold_3_epoch_29_val_loss_0.6780641078948975_accuracy_83.33333587646484.pth.tar" # test acc 66.67% on brain_seg_test_list.txt, 256x256x128 input
    # classi_sets.resume_path = "MedicalNet/dual/resnet_34_dualseq_fold_0_epoch_52_val_loss_0.4231061637401581_accuracy_91.66667175292969.pth.tar"
    # classi_sets.resume_path = "resnet_34_dualseq_fold_3_epoch_220_val_loss_0.0998634397983551_accuracy_91.66667175292969.pth.tar" # Resnet 18 test acc 69.4% on brain_seg_test_list.txt, without resize
    # classi_sets.resume_path = "convnext_50_dualseq_fold_2_epoch_63_val_loss_0.5178526639938354_accuracy_89.28572082519531.pth.tar" # ConvNext test acc 57.1% 256x256x64 with aug (trained with aug)
    classi_sets.resume_path = "convnext_50_dualseq_fold_2_epoch_70_val_loss_0.37123769521713257_accuracy_91.0714340209961.pth.tar" # ConvNext test acc 68.6% (lower than 69.4 because removed an image in testset) 256x256x64 without aug, 71.4% with aug, trained without aug
    classi_sets.data_list = "data/aug_list.txt"
    # classi_sets.data_list = "data/brain_seg_test_list_t1t2.txt"
    # classi_sets.data_list = "data/brain_seg_test_list.txt"
    #classi_sets.data_list = "MedicalNet/dual/dataset_t1_t2/data_t1_t2_match_train.txt"
    classi_sets.data_root = "data/Preprocessed"
    classi_sets.model_depth = 18 # For Resnet
    classi_sets.resnet_shortcut = 'A' # For Resnet
    classi_sets.batch_size = 1
    classi_sets.convnext_size = 'tiny'
    classi_sets.in_channels = 2
    classi_sets.gpu_id = gpu_id
    classi_sets.input_W = 256
    classi_sets.input_H = 256
    classi_sets.input_D = 64
    classi_sets.phase = 'test'
    classi_sets.export_preprocessed_image = True
    # Getting model
    checkpoint = torch.load(classi_sets.resume_path)
    model, _ = MMModel.generate_model(classi_sets)
    if classi_sets.no_cuda:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model = medcam.inject(
        model, backend='gcampp', 
        layer='auto', 
        data_shape=[256, 64, 256],
        output_dir="attention_maps", 
        save_maps=True)
    testing_data =CustomTumorDataset(classi_sets.data_root, classi_sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    results = classification(data_loader, model, classi_sets)
    for result in results[0]:
        print(result[0], result[1], "\n")
