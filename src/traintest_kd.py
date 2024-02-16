# This file is updated traintest with knowledge distillation

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
import argparse
import datetime
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.hub import download_url_to_file

import models
from utilities import *

preds_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/passt_enemble_logits_mAP_495.npy"

fname_to_index_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/fname_to_index.pkl"


def train(model, train_dataloader, test_dataloader, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device : {device}")
    torch.set_grad_enabled(True)
    # define the optimizer, loss_function, training.

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # optimization if amp is used
    optimizer.zero_grad()
    scaler = GradScaler()
    start_time = time.time()
    exp_dir = args.exp_dir
    epoch, global_step =0, 0 
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])
    model.train()

    # todo 1: read the soft labels from above files
    if not os.path.isfile(args.teachers_preds):
        print("Download teacher predictions")
        download_url_to_file(preds_url, args.teachers_preds)
    # todo 2: Kd metrics

    print(f"Load teacher predictions from file {args.teachers_preds}")
    
    # below file is for the passt 
    # teachers_preds = np.load(args.teachers_preds)


    # below file is for cnn (mobilenet) logits
    teachers_preds = np.load("/data/swarup_behera/Research/TOME/ToMe/egs/audioset/logits_balanced_train_segments.npy")
    teacher_preds = torch.from_numpy(teachers_preds).float()
    temperature = args.temperature
    teacher_preds = teacher_preds/temperature
    distillation_loss = nn.BCEWithLogitsLoss(reduction="none")
    # apply sigmoid to the result of above value
    teacher_preds = torch.sigmoid(teacher_preds)
    teacher_preds.requires_grad = False
    warmup = args.warmup
    if not os.path.isfile(args.fname_to_index):
        print("Download filename to teacher prediction index dictionary ...")
        download_url_to_file(fname_to_index_url, args.fname_to_index)
    # below code is for passt
    '''
    f = open(args.fname_to_index, 'rb')
    fname_to_index = pickle.load(f)
    '''
    # below code is for mobilenet based predictions
    fname_to_index_file_path = "/data/swarup_behera/Research/TOME/ToMe/egs/audioset/fname_to_index.pkl"
    f = open(fname_to_index_file_path, 'rb')
    fname_to_index = pickle.load(f)

   # todo 3: training loop and loss updation
    print("starting the training with knowledge distilation")
    result = np.zeros([args.n_epochs+1, 5], dtype=object)
    result[0, :] = ["acc", "mAP", "mAUC",  "valid_loss", "optimizer.param_groups[0]['lr']"]
    model.train()
    while epoch<=args.n_epochs:
        print(f"current #step={global_step}, # epochs={epoch}")
        step=0
        total_loss = 0
        for audio_input, fnames, labels in train_dataloader:
            step+=1
            audio_input = audio_input.to(device, non_blocking=True)
            # change the name to y to keep the same terminology as effientat(https://github.com/fschmid56/EfficientAT)
            y = labels.to(device, non_blocking=True)

            # first several steps for warm-up
            if global_step<=1000 and global_step%50==0 and warmup==True:
                warm_lr = (global_step/1000) * args.lr
                for params in optimizer.param_groups:
                    params['lr'] = warm_lr
            # fetch the correct index in 'teacher_preds' for given filename
                # insert -1 for files not in fname_to_index (proportion of files successfully downloaded from
                # YouTube can vary for AudioSet)
            '''
            indices = torch.tensor(
                [fname_to_index[fname] if fname in fname_to_index else -1 for fname in fnames], dtype=torch.int64
            )
            '''
            # Initialize an empty list to store the indices
            indices_list = []

            for fname in fnames:
                fname = fname.replace(".wav", "").split("/")[-1][1:]
                # Check if the filename exists in the fname_to_index dictionary
                if fname in fname_to_index:
                        # If yes, append the corresponding index to the list
                        indices_list.append(fname_to_index[fname])
                else:
                        # If no, append -1 to the list
                        indices_list.append(-1)

            # Convert the list of indices to a PyTorch tensor of type torch.int64
            indices = torch.tensor(indices_list, dtype=torch.int64)
            # get indices of files we could not find the teacher predictions for
            unknown_indices = indices == -1
            y_soft_teacher = teacher_preds[indices]
            
            
            # TODO:  convert above code to batch format and continue from here and test the changes of dataloader.

            with autocast():
                y_hat= model(audio_input)
                loss_func = nn.BCEWithLogitsLoss()
                y_soft_teacher = y_soft_teacher.to(y_hat.device).type_as(y_hat)
                samples_loss = loss_func(y_hat, y)
                # hard label loss
                label_loss = samples_loss.mean()

                y_soft_teacher = teacher_preds[indices]
                y_soft_teacher = y_soft_teacher.to(y_hat.device).type_as(y_hat)

                soft_targets_loss = distillation_loss(y_hat, y_soft_teacher)


                soft_targets_loss[unknown_indices] = soft_targets_loss[unknown_indices] * 0
                soft_targets_loss = soft_targets_loss.mean()

                # weighting losses
                label_loss = args.kd_lambda * label_loss
                soft_targets_loss = (1 - args.kd_lambda) * soft_targets_loss

                # total loss is sum of lambda-weighted label and distillation loss
                total_loss = label_loss + soft_targets_loss
                
                # Update Model

                # I need to find differnet metrics after each epoch.
                # Also print validation scores of above metrics after each score.
                # Print time of kdtome and simple kd with different values of r.

                
            global_step+=1
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
        print(f"loss: {total_loss}")
        

        print("Starting validation ...............")
        stats, valid_loss = validation(model, test_dataloader, args, epoch)
        print(f"validation loss: {valid_loss}")

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        print("mAP: {:.6f}".format(mAP))
        print("acc: {:.6f}".format(acc))

        result[epoch, :] = [acc, mAP, mAUC,  valid_loss,  optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        epoch+=1


def validation(model,test_dataloader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    # switch to evaluate mode
    model.eval()

    A_predictions = []
    A_targets = []
    A_loss = []
    
    with torch.no_grad():
        for audio_inputs, _ , labels in test_dataloader:
            audio_inputs= audio_inputs.to(device)

            # compute the output of model
            output_logits = model(audio_inputs)
            output = torch.sigmoid(output_logits)
            predictions = output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # calculae the loss
            labels = labels.to(device)
            loss_func = nn.BCEWithLogitsLoss()
            if isinstance(loss_func, nn.CrossEntropyLoss):
                loss = loss_func(output,  np.argmax(labels.long(), axis=1))
            else: 
                loss = loss_func(output,labels)
            A_loss.append(loss.to('cpu').detach()) 
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

    return stats, loss





if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'input args')
    parser.add_argument("--lr", default=0.5, help="learning rate")
    parser.add_argument("--n_epochs", default=8, help="number of epochs")
    parser.add_argument("--exp_dir", default="./exp_1", help="experiment directory")
    args = parser.parse_args()

    input_tdim = 100
    ast_mdl = models.ASTModel(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    train(ast_mdl, " ", " ", args)
    '''
    input_tdim = 256
    ast_mdl = models.ASTModel(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    '''
