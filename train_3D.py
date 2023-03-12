import argparse
import os
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
import loss as L
import pandas as pd
from model import Optim_U_Net_3d
from data import normal, correct, cs_data, cs_gt_data, flip
from tool import IOUDICE

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to load image and label dataset')
parser.add_argument('--saveroot', required=True, help='path to save image and label dataset')
parser.add_argument('--inference_dataset', required=True, help='path to C-scan cross-sectional image dataset')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--load_model', type=bool, default=False, help='load pre-trainined model to fine-tune')
parser.add_argument('--modelpath', type=str, default="your model path", help='the model path to load')
parser.add_argument('--lr', type=float, default="0.001", help='learning rate')
parser.add_argument('--step', type=int, default="15", help='step size of scheduler')
parser.add_argument('--epoch', type=int, default="50", help='number of epochs to train')
opts = parser.parse_args()

def train_3d(opt): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Optim_U_Net_3d(img_ch=opt.input_nc,output_ch=opt.output_nc)
    if(opt.load_model):
        model.load_state_dict(torch.load(opt.modelpath))
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(number_of_parameters)
    
    train_opath = opt.dataroot + '/train/'
    test_opath = opt.dataroot + '/test/'
    train_list = sorted(os.listdir(train_opath))
    test_list = sorted(os.listdir(test_opath))
    
    gpath = opt.gt_dataroot + '/'
    
    loss_func = L.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)
    
    train_loss=np.zeros(opt.epoch)
    valid_loss=np.zeros(opt.epoch)
    train_mdice=np.zeros(opt.epoch)
    valid_mdice=np.zeros(opt.epoch)
    
    depth = 384
    height = 1024
    width = 1024
    
    cube = 128
    
    a = int(height/cube)
    b = int(width/cube)

    TRAIN_img_sub = np.zeros((2,1,1,depth,height,width))
    TRAIN_gim_sub = np.zeros((2,1,1,depth,height,width))
    k = 0
    for filename in train_list:
        TRAIN_img_sub[k] = normal(correct(cs_data(train_opath, filename)),2,98)
        TRAIN_gim_sub[k] = cs_gt_data(gpath,filename, depth, height, width)
        k+=1
        print("Train ready")
    for filename in test_list:
        TEST_img_sub = normal(correct(cs_data(test_opath, filename)),2,98)
        TEST_gim_sub = cs_gt_data(gpath,filename, depth, height, width)
        print("Test ready")
    loss = 0
    for EPOCH in range(opt.epoch):
        print('\nEPOCH:{} learning rate:{}====================================='.format(EPOCH,optimizer.param_groups[0]['lr']))
        start = time.time()
        model.train()
        trainloss_list = np.empty((0,1))
        train_dice_list = np.empty((0,1))
        order = np.arange(a*b)
        vol = np.random.randint(2,size=len(order))
        np.random.shuffle(order)
        option = np.random.randint(4,size=len(order))
        i=0
        for odr in order:
            yy = (odr // b) * cube
            xx = (odr % b) * cube
            img_sub = flip(TRAIN_img_sub[vol[i],:,:,:,xx:xx+cube,yy:yy+cube], option[i])
            gim_sub = flip(TRAIN_gim_sub[vol[i],:,:,:,xx:xx+cube,yy:yy+cube], option[i])
            INPUT =  torch.from_numpy(img_sub.astype(np.float32)).to(device = device, dtype = torch.float)
            OUTPUT = model(INPUT)
            loss = loss_func(OUTPUT, L.make_one_hot(torch.from_numpy(gim_sub.copy()).to(dtype = torch.int64),2).cuda())
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            trainloss_list = np.vstack((trainloss_list,loss.item()))
            train_dice = np.array(IOUDICE(np.argmax(OUTPUT.cpu().detach().numpy(), 1),gim_sub.squeeze(1),1))
            train_dice_list = np.vstack((train_dice_list,train_dice.item())) 
            i+=1
        model.eval()
        validloss_list = np.empty((0,1))
        val_dice_list  = np.empty((0,1))
        for filename in test_list:
            order = np.arange(a*b)
            for odr in order:
                yy = (odr // b) * cube
                xx = (odr % b) * cube
                INPUT =  torch.from_numpy(np.array(TEST_img_sub[:,:,:,xx:xx+cube,yy:yy+cube]).astype(np.float32)).to(device = device, dtype = torch.float)
                OUTPUT = model(INPUT)
                loss = loss_func(OUTPUT, L.make_one_hot(torch.from_numpy(TEST_gim_sub[:,:,:,xx:xx+cube,yy:yy+cube]).to(dtype = torch.int64),2).cuda())
                validloss_list = np.vstack((validloss_list, loss.item()))
                val_dice = np.array(IOUDICE(np.argmax(OUTPUT.cpu().detach().numpy(),1),TEST_gim_sub[:,:,:,xx:xx+cube,yy:yy+cube].squeeze(1),1))
                val_dice_list = np.vstack((val_dice_list,val_dice.item()))
        scheduler.step()
        train_loss[EPOCH],valid_loss[EPOCH] = trainloss_list.mean(),validloss_list.mean()
        train_mdice[EPOCH],valid_mdice[EPOCH] = train_dice_list.mean(),val_dice_list.mean()
        print('%.3d'%EPOCH, "train loss:", '%.3f'%trainloss_list.mean(), "valid loss:", '%.3f'%validloss_list.mean(),
              "train DICE:",'%.3f'%train_dice_list.mean(), "valid DICE:",'%.3f'%val_dice_list.mean(),round((time.time()-start),3))
    # save last epoch
    torch.save(model.state_dict() ,'{}.pkl'.format(opt.name))
    print("----saving successfully with name: {}.pkl----".format(opt.name))

    columns = ['training loss', "validation loss", 
    'training DICE (cell)', "validation DICE (cell)"]
    df = pd.DataFrame([train_loss,valid_loss,train_mdice,valid_mdice],columns=columns)
    df.to_csv(opt.name+"3D_cell_segmentation"+opt.name+".csv",index=False)
    return 

train_3d(opts)