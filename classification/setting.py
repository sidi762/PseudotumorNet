'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default='./data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--data_root_val',
        default='./data',
        type=str,
        help='Root directory path of the validation set')
    parser.add_argument(
        '--img_list',
        #default='./data/train.txt',
        default='',
        type=str,
        help='Path for image list file')
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.01,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
        default=20,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=512,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=512,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        #default='pretrain/resnet_50.pth',
        default = '',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        #default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['conv_seg'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument(
        '--with_augmentation', 
        action='store_true', 
        help='If true, data after brightness augmentation is used.')
    parser.add_argument(
        '--k_fold_cross_validation', 
        action='store_true', 
        help='If true, k-fold cv is used.')
    parser.set_defaults(no_cuda=False)
    parser.set_defaults(with_augmentation=False)
    parser.set_defaults(k_fold_cross_validation=False)
    parser.add_argument(
        '--export_preprocessed_image', 
        action='store_true', 
        help='If true, image after preprocessing will be saved.')
    parser.set_defaults(export_preprocessed_image=False)
    parser.add_argument(
        '--preprocessed_image_export_path', 
        type=str,
        default='./preprocessed_image',
        help='Path for exporting preprocessed images.')
    parser.add_argument(
        '--manual_train_val_split', 
        action='store_true', 
        help='If true, manual train-test split is used.')
    parser.set_defaults(manual_train_val_split=False)
    parser.add_argument(
        '--train_ids',
        default=[],
        type=list,
        help='images for train set, only valid when manual_train_val_split is set to True')
    parser.add_argument(
        '--val_ids',
        default=[],
        type=list,
        help='images for train set, only valid when manual_train_val_split is set to True')
    parser.add_argument(
        '--fold_num',
        default='5',
        type=int,
        help='fold number for k-fold cv')
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--convnext_size',
        default='base',
        type=str,
        help='(base | tiny | large | small | xlarge | ')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--in_channels',
        default=2,
        type=int,
        help='Number of input channels')
    parser.add_argument(
        '--manual_seed', default=42, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}".format(args.model, args.model_depth)

    return args
