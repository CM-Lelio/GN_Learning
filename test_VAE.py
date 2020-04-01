# -.- encoding: utf-8 -.-
import torch
import os,sys
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import numpy as np
import lmdb
import argparse
import matplotlib.pyplot as plt
from scipy.misc import imresize
import random

from loader import myDataset
from Myutils import *

## import models from [model] dir
from Models import *

parser = argparse.ArgumentParser()
parser.add_argument("--loadModel", default='none', type=str, help="reload feature extractor")
parser.add_argument("--logMark", default='none', type=str, help="the log file name mark")
parser.add_argument("--model", default='AlexDiff', type=str, help="the model will train")
parser.add_argument("--valueBatch", default=1, type=int, help="the batch size of valuation")
parser.add_argument("--lossFunc", default='CEloss', type=str, help="loss function")
parser.add_argument("--GPU", default=1, type=int, help="flag for using GPUs")
args = parser.parse_args()

## some preparation
savePath = './SaveModel/'
if os.path.exists(savePath) != True:
    os.mkdir(savePath)

if os.path.exists('./Testlog/') != True:
    os.mkdir('./Testlog/')

logf = logFile('./Testlog/'+args.logMark)
logf.logging('------------Head Information----------')


valuePath='data/test/'
valuelabelPath='data/t10k-labels-idx1-ubyte.gz'

value_data = myDataset(valuelabelPath,valuePath)
value_loader = DataLoader(dataset=value_data, batch_size=args.valueBatch, shuffle=False, num_workers=3)

logf.logging("Value batch size: " + str(args.valueBatch))

## model and loss function
if args.GPU == 1:
    modeleval = args.model+'().cuda()'
else:
    modeleval = args.model+'()'
model=eval(modeleval)
print(model)
logf.logging("build model: model/" + args.model)
    
if args.lossFunc == 'CEloss':
    if args.GPU == 1:
        loss_func = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
else:
    print('there is an error in loss function!')
    os._exit(0)

logf.logging("loss function: " + args.lossFunc)

epochSaved =0 
if args.loadModel != 'none':
    model.load_state_dict(torch.load(savePath+args.loadModel))
    logf.logging('load model parameter: '+savePath+args.loadModel)

logf.logging('------------Head End----------')

## main loop
for epoch in range(epochSaved,epochSaved+1):

    # valuation loop
    model.eval()
    value_loss=0
    value_acc=0
    barcount=0
    loss_count=0
    fig, [ax1, ax2, ax3]=plt.subplots(1,3)
    for batch_x,batch_y in value_loader:
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y.type('torch.LongTensor'))
        if args.GPU == 1:
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
        output,attn_map=model(batch_x,True)
        loss = loss_func(output,batch_y[:,0])
        value_loss += loss.data.item()
        #print(loss.data[0])

        i=random.randint(0,attn_map.size(0)-1)
        attn=attn_map.cpu().data[i,0,:,:].numpy()
        ax1.imshow(attn,cmap=plt.cm.gray)
        img=batch_x.cpu().data[i,0,:,:].numpy()
        ax2.imshow(img,cmap=plt.cm.gray)
        attn=imresize(attn,[28,28])
        maskimg=img*(attn/255)
        ax3.imshow(maskimg,cmap=plt.cm.gray)
        plt.pause(0.001)
        #assert(1==2)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        value_acc += pred.eq(batch_y.view_as(pred)).sum().item()

        progress_bar(barcount,len(value_data))
        barcount+=batch_x.size(0)
        loss_count=loss_count+1

    print('')
    logf.logging('Epoch: {}, Value Loss: {:.7f}'.format(epoch+1,value_loss / loss_count))
    logf.logging('Value samples: {}, Value Acc: {:.4f}'.format(len(value_data),value_acc / (len(value_data))))

    logf.logging('-------------------------------------------------------------')
