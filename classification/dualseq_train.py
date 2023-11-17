from setting import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
from datasets.wch_dataset import CustomTumorDataset
from torch.utils.tensorboard import SummaryWriter
from EarlyStopping_torch import EarlyStopping
from sklearn.model_selection import KFold
from datetime import datetime
import schedulers


def train(data_loader, validation_loader, model, total_epochs, 
          save_folder, sets, patience, 
          fold=None, optimizer=None, scheduler=None, 
          lr_schedule_values=None, update_freq=None, save_interval=None):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_func = nn.CrossEntropyLoss()
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)

    log.info("Current setting is:")
    log.info(sets)
    log.info("\n\n")
    if not sets.no_cuda:
        loss_func = loss_func.cuda()

    model.train(True)
    train_time_sp = time.time()
    best_val_loss = 1000
    val_accuracy = 0 # Store the validation accuracy for this fold
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            log.info('lr = {}'.format(current_lr))
            writer.add_scalar("LearningRate", current_lr, epoch)
        else:
            current_lr = lr_schedule_values[epoch * batches_per_epoch] 
            log.info('lr = {}'.format(current_lr))
            writer.add_scalar("LearningRate", current_lr, epoch)

        for batch_id, batch_data in enumerate(data_loader):
            if epoch == 0 and batch_id == 0:
                writer.add_graph(model, input_to_model=batch_data[0], verbose=False)
            correct = 0
            total = 0
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[batch_id + batch_id_sp]
                        log.info('lr = {}'.format(param_group["lr"]))
            
            volumes, label, img_name = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_class = model(volumes)
            if not sets.no_cuda:
                out_class = out_class.cuda()
                label = label.cuda()

            # calculating loss
            loss = loss_func(out_class, label)
            loss.backward()
            optimizer.step()
            if scheduler is not None: 
                scheduler.step()
            last_loss = loss.item()
            _, predicted = torch.max(out_class.data, 1)
            correct += (predicted == label).float().sum()
            total += label.size(0)
            accuracy = 100 * correct / total
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, accuracy = {:.3f} avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, last_loss, accuracy, avg_batch_time))

        #Validation per epoch
        with torch.no_grad():
            model.train(False)
            correct = 0
            total = 0
            running_val_loss = 0.0
            for batch_id, batch_data in enumerate(validation_loader):
                batch_id_sp = epoch * batches_per_epoch
                val_volumes, val_labels, val_img_names = batch_data

                if not sets.no_cuda:
                    val_volumes = val_volumes.cuda()

                val_out_class = model(val_volumes)
                if not sets.no_cuda:
                    val_out_class = val_out_class.cuda()
                    val_labels = val_labels.cuda()

                _, predicted = torch.max(val_out_class.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).float().sum()
                #Printing the ones that the model failed to predict
                for index, item in enumerate(predicted):
                    if item != val_labels[index]:
                        log.info("{} should be {}".format(val_img_names[index], val_labels[index]))

                val_loss = loss_func(val_out_class, val_labels)
                running_val_loss += val_loss


            val_accuracy = 100 * (correct / total)
            avg_val_loss = running_val_loss / (batch_id + 1)
            log.info('Validation loss {}'.format(avg_val_loss))
            log.info('Validation accuracy {}'.format(val_accuracy))
            writer.add_scalars("Training vs. Validation Loss", {'Train': last_loss, 'Validation': avg_val_loss}, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            if (avg_val_loss < best_val_loss) or (val_accuracy >= 75.0):
                best_val_loss = avg_val_loss
                model_save_path = '{}_dualseq_fold_{}_epoch_{}_val_loss_{}_accuracy_{}.pth.tar'.format(save_folder, fold, epoch, avg_val_loss, val_accuracy)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                log.info('Saved checkpoints: fold = {} epoch = {} avg_val_loss = {} accuracy = {}'.format(fold, epoch, avg_val_loss, val_accuracy))
                torch.save({'epoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            model_save_path)

            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                log.info("Early stopping")
                break
        #End Validation

    
    writer.flush()
    log.info('Finished training')
    if sets.ci_test:
        exit()
    return val_accuracy


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    
    # Commented out as this is set in arguments 
    # sets.model = 'resnet'
    # sets.model_depth = 34
    # sets.resnet_shortcut = 'A'
    
    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
        
    #EarlyStopping
    patience = 300
    
    # Set fixed random number seed
    torch.manual_seed(sets.manual_seed)
    
    # full_dataset = CustomTumorDataset(sets.data_root, sets)
    # k_folds = 5
    # kfold = KFold(n_splits=k_folds, shuffle=True)
    dataset_train = CustomTumorDataset(sets.data_root, sets)
    dataset_val = CustomTumorDataset(sets.data_root_val, sets)
    print('Training set has {} instances'.format(len(dataset_train)))
    print('Validation set has {} instances'.format(len(dataset_val)))
    
    tb_log_dir = ""
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    if sets.model == 'resnet':
        tb_log_dir = "runs/"+sets.model+str(sets.model_depth)+"_"+date_time
    elif sets.model == 'convnext':
        tb_log_dir = "runs/"+sets.model+sets.convnext_size+"_"+date_time
    
    writer = SummaryWriter(log_dir=tb_log_dir)
    model, parameters = generate_model(sets) # Generate Models
          
    log.info (model)
    # Compile model for faster training
    # model = torch.compile(model)
    
    # optimizer
    if (not sets.ci_test) and sets.pretrain_path:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate },
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
        ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    # optimizer = torch.optim.Adam(params,
    #                              lr=sets.learning_rate,
    #                              betas=(0.9,0.999),
    #                              eps=1e-08,
    #                              weight_decay=1e-3,
    #                              amsgrad=False)
    # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.AdamW(params, eps=1e-8, lr=sets.learning_rate, weight_decay=0.05)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    print("Use Cosine LR scheduler")
    num_training_steps_per_epoch = (len(dataset_train) // sets.batch_size) + 1
    print("num_training_steps_per_epoch: " + str(num_training_steps_per_epoch))
    lr_schedule_values = schedulers.cosine_scheduler(
        sets.learning_rate, 1e-6, sets.n_epochs, num_training_steps_per_epoch,
        warmup_epochs=20, warmup_steps=-1,
    )
    
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                     T_0 = 20,# Number of iterations for the first restart
    #                                     T_mult = 1, # A factor increases TiTi​ after a restart
    #                                     eta_min = 1e-6) # Minimum learning rate
    
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                     T_0 = sets.n_epochs,# Number of iterations for the first restart
    #                                     T_mult = 1, # A factor increases TiTi​ after a restart
    #                                     eta_min = 1e-6) # Minimum learning rate
    
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=sets.n_epochs, eta_min=1e-6)

       
        
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            log.info("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
            .format(sets.resume_path, checkpoint['epoch']))

    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_data_loader = DataLoader(dataset_train, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    val_data_loader = DataLoader(dataset_val, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
       
    train(data_loader=train_data_loader, validation_loader=val_data_loader, 
          model=model, optimizer=optimizer, total_epochs=sets.n_epochs, 
          save_interval=sets.save_intervals, save_folder=sets.save_folder, 
          sets=sets, patience=patience, lr_schedule_values=lr_schedule_values)   
  
    
