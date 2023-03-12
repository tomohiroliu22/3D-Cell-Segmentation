import os
import random
import numpy as np
import cv2
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset



def data_loader(opath, fold):
    """ Data loader to split training, validation, and testing set.
    INPUT:
        [opath]: path to your image files directory (ex: ./[your own path]/dataset/dataset/image/)
        [fold]: selcet the fold for cross-validation
    OUTPUT:
        [train_data_LD]: training set data loader
        [valid_data_LD]: validation set data loader
    """
    n = int(len(os.listdir(opath))/5) 
    all_data_filename = sorted(os.listdir(opath))
    random.Random(42).shuffle(all_data_filename)   # seed = 4
    fold_list = [all_data_filename[i:i+n] for i in range(0,len(all_data_filename),n)]
    #　valid data
    valid_data_list = fold_list[fold]
    # train data
    train_data_list = []
    [train_data_list.append(x) for x in all_data_filename if x not in (valid_data_list)]
    
    print('train',len(train_data_list),'valid',len(valid_data_list))
    
    train_data_LD = Data.DataLoader(dataset=train_data_list, batch_size=8, shuffle=True, num_workers=4)
    valid_data_LD = Data.DataLoader(dataset=valid_data_list, batch_size=8, shuffle=False, num_workers=4)
    return  train_data_LD, valid_data_LD

def data_loader_semi(opath, opath2, fold):
    n = int(len(os.listdir(opath))/5) 
    data_filename = sorted(os.listdir(opath))
    random.Random(42).shuffle(data_filename) 
    data_filename2 = sorted(os.listdir(opath2))
    random.Random(42).shuffle(data_filename2) 
    all_data_filename = data_filename+data_filename2
    random.Random(42).shuffle(all_data_filename) 
    fold_list = [data_filename[i:i+n] for i in range(0,len(data_filename),n)]
    #　valid data
    valid_data_list = fold_list[fold]
    # train data
    train_data_list = []
    [train_data_list.append(x) for x in all_data_filename if x not in (valid_data_list)]
    
    print('train',len(train_data_list),'valid',len(valid_data_list))
    
    train_data_LD = Data.DataLoader(dataset=train_data_list, batch_size=8, shuffle=True, num_workers=4)
    valid_data_LD = Data.DataLoader(dataset=valid_data_list, batch_size=8, shuffle=False, num_workers=4)
    return  train_data_LD, valid_data_LD


def o_data(opath, batch_num, width, height):
    """ reading OCT images
    INPUT:
        [opath]: path to your image files directory (ex: ./[your own path]/dataset/dataset/image/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
    OUTPUT:
        [img_sub]: OCT images batch 
    """
    img_sub = np.zeros((len(batch_num),1,width,height))
    i = 0
    for name in np.array(batch_num):
        img_sub[i,0,:,6:506] = cv2.imread(opath + name, 0)/255
        i+=1 
    return img_sub

def o_data_semi(opath,opath2,batch_num,width,height):
    img_sub = np.zeros((len(batch_num),1,width,height))
    i = 0
    for name in np.array(batch_num):
        if(name[0]=='v' or name[0]=='h'):
            img_sub[i,0,:,6:506] = cv2.imread(opath2 + name, 0)/255
        else:
            img_sub[i,0,:,6:506] = cv2.imread(opath + name, 0)/255
        i+=1 
    return img_sub

def g_data_cell(gpath, batch_num,width,height):
    """ reading cell labeling for deep feature sharing model
    INPUT:
        [gpath]: path to your cell labeling files directory (ex: ./[your own path]/dataset/dataset/cell/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
    OUTPUT:
        [img_sub]: OCT images batch 
        [img_sub2]: OCT images batch for down-sampling
    """
    NUM = len(batch_num)
    img_sub = np.zeros((NUM,1,width,height))
    half_width = int(width/2)
    half_height = int(height/2)
    img_sub2 = np.zeros((NUM,1,half_width,half_height))
    i = 0
    for name in np.array(batch_num):
        img = cv2.imread(gpath + name, 0)/255
        img_sub[i,0,:,6:506] = img
        img_half = cv2.resize(img,(250,half_width),interpolation=cv2.INTER_NEAREST)
        img_sub2[i,0,:,3:253] = img_half
        i+=1
    return img_sub, img_sub2

def g_data_cell_semi(gpath,gpath2,batch_num,width,height):
    NUM = len(batch_num)
    half_width = int(width/2)
    half_height = int(height/2)
    img_sub = np.zeros((NUM,1,width,height))
    img_sub2 = np.zeros((NUM,1,half_width,half_height))
    i = 0
    for name in np.array(batch_num):
        if(name[0]=='v' or name[0]=='h'):
            img = cv2.imread(gpath2 + name, 0)/255
        else:
            img = cv2.imread(gpath + name, 0)/255
        img_half = cv2.resize(img,(250,half_width),interpolation=cv2.INTER_NEAREST)
        img_sub[i,0,:,6:506] = img
        img_sub2[i,0,:,3:253] = img_half
        i+=1
    return img_sub, img_sub2

def inf_data_loader(opath):
    all_data_filename = sorted(os.listdir(opath))
    data_LD = Data.DataLoader(dataset=all_data_filename, batch_size=8, shuffle=False, num_workers=4)
    return  data_LD

def cs_data(path, filename):
    img = np.zeros((1,1,384,1000,1000))
    img[0,0] = np.fromfile(path + filename, dtype='float32', sep="").reshape(384,1000,1000)
    img_temp = np.zeros((1,1,384,1000,1000))
    for i in range(1000):
        if((i>2) and (i<988)):
            img_temp[0,0,:,i,:] = (img[0,0,:,i-1,:]+img[0,0,:,i,:]+img[0,0,:,i+1,:])/3
        else:
            img_temp[0,0,:,i,:] = img[0,0,:,i,:]
    img_temp2 = np.zeros((1,1,384,1000,1000))
    for i in range(1000):
        if((i>2) and (i<988)):
            img_temp2[0,0,:,:,i] = (img[0,0,:,:,i-1]+img[0,0,:,:,i]+img[0,0,:,:,i+1])/3
        else:
            img_temp2[0,0,:,:,i] = img[0,0,:,:,i]
    img = (img_temp+img_temp2)/2
    del img_temp
    del img_temp2
    return img

def cs_gt_data(gpath,filename, depth, height, width):
    img = np.zeros((1,1,depth,width,height))
    img[0,0,:,12:1012,12:1012] = np.fromfile(gpath + filename, dtype='float32', sep="").reshape(384,1000,1000)
    img = (img*255).astype(np.int_)
    return img/255

def normal(data,MIN,MAX):
    p,q,x,y,z=data.shape
    data=data.flatten()
    vmin,vmax=np.percentile(data,(MIN,MAX))
    data[data<vmin]=vmin
    data[data>vmax]=vmax
    data=(data-vmin)/(vmax-vmin)
    data=data.reshape(p,q,x,y,z)
    data_final = np.zeros((p,q,x,1024,1024))
    data_final[0,0,:,12:1012,12:1012] = data
    return data_final

def correct(img):
    y = np.zeros(384)
    for i in range(384):
        y[i] = np.mean(img[0,0,i].ravel())
    x = np.arange(384)/384
    z = np.polyfit(x, y, 12)
    p = np.poly1d(z)
    y_pred = p(x)
    for i in range(384):
        img[0,0,i] = img[0,0,i] + (y_pred[i]-y[i])
    return img

def flip(img,option):
    if(option==0):
        return img
    elif(option==1):
        return img[:,:,:,:,::-1]
    elif(option==2):
        return img[:,:,:,::-1,:]
    else:
        return img[:,:,:,::-1,::-1]
