import torch
from torch import nn
import sys
from models_def import resnet, convnext3d


def generate_model(opt):
    assert opt.model in [
        'resnet',
        'convnext'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                in_channel=opt.in_channels,
                num_seg_classes=opt.n_seg_classes)
    elif opt.model == 'convnext':
        assert opt.convnext_size in ['base', 'large', 'small', 'tiny', 'xlarge']
        if opt.convnext_size == 'base':
            model = convnext3d.convnext3d_base(in_chans=opt.in_channels)
        elif opt.convnext_size == 'large':
            model = convnext3d.convnext3d_large(in_chans=opt.in_channels)
        elif opt.convnext_size == 'small':
            model = convnext3d.convnext3d_small(in_chans=opt.in_channels)
        elif opt.convnext_size == 'tiny':
            model = convnext3d.convnext3d_tiny(in_chans=opt.in_channels)
        elif opt.convnext_size == 'xlarge':
            model = convnext3d.convnext3d_xlarge(in_chans=opt.in_channels)

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
        pretrain['state_dict'].pop('module.conv1.weight')
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict, strict=False)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
