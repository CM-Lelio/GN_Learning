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
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

from loader import MyDataset
from Myutils import *

## import models from [model] dir
from VAE_module import *

parser = argparse.ArgumentParser()
parser.add_argument("--loadModel", default='none', type=str, help="reload feature extractor")
parser.add_argument("--logMark", default='none', type=str, help="the log file name mark")
parser.add_argument("--maxEpoch", default=50, type=int, help="the max epoch for training")
parser.add_argument("--model", default='AlexDiff', type=str, help="the model will train")
parser.add_argument("--trainBatch", default=64, type=int, help="the batch size of training")
parser.add_argument("--valueBatch", default=64, type=int, help="the batch size of valuation")
parser.add_argument("--save", default='none', type=str, help="dir name of saving mode")
parser.add_argument("--saveEpoch", default=1, type=int, help="each epoch to save")
parser.add_argument("--lossFunc", default='CEloss', type=str, help="loss function")
parser.add_argument("--optim", default='Adam', type=str, help="optimization")
parser.add_argument("--learnRate", default=0.1, type=float, help="learning rate for training")
parser.add_argument("--Momentum", default=0.9, type=float, help="momentum for SGD")
parser.add_argument("--weightDecay", default=0, type=float, help="weight_decay for SGD")
parser.add_argument("--LRS", default=10, type=int, help="Learning rate schedule for SGD")
parser.add_argument("--GPU", default=0, type=int, help="flag for using GPUs")
args = parser.parse_args()

def drawfig(epochs, trloss, tsloss, fig_name):
    fig = plt.figure()
    plt.semilogy(np.array(epochs),np.array(trloss),'-.',label='train loss')
    plt.semilogy(np.array(epochs),np.array(tsloss),'-^',label='test loss')
    ax = plt.gca()
    ax.xaxis.set_major_locator(
        plt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(
        plt.MultipleLocator(0.1))
    plt.grid(which='major', axis='both',
        linewidth=0.75, linestyle='-',
        color='lightgray')
    plt.grid(which='minor', axis='both',
        linewidth=0.25, linestyle='-',
        color='lightgray')
    plt.xlabel("epoch") 
    plt.ylabel("loss value")
    plt.legend()
    plt.savefig(fig_name)
    plt.close(fig)

## some preparation
savePath = './SaveModel/'
if os.path.exists(savePath) != True:
    os.mkdir(savePath)

if args.save != 'none' and os.path.exists(savePath+args.save) != True:
    os.mkdir(savePath+args.save)

if os.path.exists('./log/') != True:
    os.mkdir('./log/')

logf = logFile('./log/'+args.logMark)
logf.logging('------------Head Information----------')


root="../fashion-mnist/"

train_data = MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

logf.logging("Train batch size: " + str(args.trainBatch))
logf.logging("Value batch size: " + str(args.valueBatch))

## model and loss function
if args.GPU == 1:
    modeleval = args.model+'().cuda()'
else:
    modeleval = args.model+'()'
model=eval(modeleval)
print(model)
logf.logging("build model: model/" + args.model)
logf.logging("optim: {}".format(args.optim))
logf.logging("learning rate: {:6f}".format(args.learnRate))
if args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnRate)
elif args.optim == 'AdeDelta':
    optimizer = torch.optim.Adadelta(model.parameters())
elif args.optim == 'SGD':
    LR = args.learnRate
    M = args.Momentum
    WD = args.weightDecay
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=M, weight_decay=WD)
else:
    print('there is an error in optim!')
    os._exit(0)
    
if args.lossFunc == 'MSELoss':
    if args.GPU == 1:
        loss_func = torch.nn.MSELoss().cuda()
    else:
        loss_func = torch.nn.MSELoss()
elif args.lossFunc == 'L1Loss':
    if args.GPU == 1:
        loss_func = torch.nn.L1Loss().cuda()
    else:
        loss_func = torch.nn.L1Loss()
elif args.lossFunc == 'BCELoss':
    if args.GPU == 1:
        loss_func = torch.nn.BCELoss(reduction='sum').cuda()
    else:
        loss_func = torch.nn.BCELoss(reduction='sum')
else:
    print('there is an error in loss function!')
    os._exit(0)

logf.logging("loss function: " + args.lossFunc)
logf.logging("optim: " + args.optim)
if args.optim == 'SGD':
    logf.logging(("learning rate: {}").format(args.learnRate))
    logf.logging(("Momentum: {}").format(args.Momentum))
    logf.logging(("weight decay: {}").format(args.weightDecay))

epochSaved =0 
if args.loadModel != 'none':
    model.load_state_dict(torch.load(savePath+args.loadModel))
    logf.logging('load model parameter: '+savePath+args.loadModel)
    try:
        epochSaved = args.loadModel.split(".")[0]
        epochSaved = epochSaved.split("epoch")[-1]
        epochSaved = int(epochSaved)
    except:
        epochSaved = 0

logf.logging('------------Head End----------')

if epochSaved >= args.maxEpoch:
    print('there is an error in maxEpoch')
    os._exit(0)

## preparing for plot
drawLen = 10
tmptrloss=[]
tmptsloss=[]
tmpepochs=[]
trloss=[]
tsloss=[]
epochs=[]

showFlag=False
drawLine=True

## main loop
for epoch in range(epochSaved,args.maxEpoch):

    # training loop
    train_loss=0
    barcount=0
    loss_count=0
    # switch to train mode
    model.train()

    if showFlag:
        fig = plt.figure(figsize=(2, 1))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

    for batch_x,batch_y in train_loader:
        optimizer.zero_grad()
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y.type('torch.LongTensor'))
        if args.GPU == 1:
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
        output, kl_loss = model(batch_x)
        loss = loss_func(output.view(output.size()[0],-1), batch_x.view(batch_x.size()[0],-1)) / output.size(0) + kl_loss
        train_loss += loss.data.item()

        loss.backward()
        optimizer.step()
        progress_bar(barcount,len(train_data))
        barcount+=batch_x.size(0)
        loss_count=loss_count+1
        if showFlag:
            sample_in = batch_x[0].cpu().data.numpy()
            sample_out = output[0].cpu().data.numpy()
            plt.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            ax1.imshow(sample_in.reshape(28,28), cmap='Greys_r')
            plt.axis('off')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_aspect('equal')
            ax2.imshow(sample_out.reshape(28,28), cmap='Greys_r')
            plt.pause(0.001)
            print(abs(sample_out-sample_in).mean())

    if showFlag:
        plt.close(fig)

    print('')
    logf.logging('Epoch: {}, Train Loss: {:.7f}'.format(epoch+1,train_loss / loss_count))
    
    trloss.append(train_loss / loss_count)
    if len(tmptrloss) < drawLen:
        tmptrloss.append(train_loss / loss_count)
    else:
        tmptrloss.pop(0)
        tmptrloss.append(train_loss / loss_count)
    ## save model
    if args.save != 'none' and epoch%args.saveEpoch == 0:
        torch.save(model.state_dict(),savePath+args.save+'/Para_epoch{}.pkl'.format(epoch+1))
        print('saved model in '+ savePath+args.save)
    
    # valuation loop
    model.eval()
    value_loss=0
    barcount=0
    loss_count=0

    isSampled = False
    sample_in = []
    sample_out = []

    for batch_x,batch_y in test_loader:
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y.type('torch.LongTensor'))
        if args.GPU == 1:
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
        output, kl_loss = model(batch_x)
        loss = loss_func(output.view(output.size()[0],-1), batch_x.view(batch_x.size()[0],-1)) / output.size(0) + kl_loss
        value_loss += loss.data.item()

        progress_bar(barcount,len(test_data))
        barcount+=batch_x.size(0)
        loss_count=loss_count+1
        if isSampled == False:
            isSampled = True
            if batch_x.size(0) >= 16:
                sample_in = batch_x[:16].cpu()
                sample_in = sample_in.data.numpy().copy()
                sample_out = output[:16].cpu()
                sample_out = sample_out.data.numpy().copy()
            else:
                isSampled = False

    print('')
    logf.logging('Epoch: {}, Value Loss: {:.7f}'.format(epoch+1,value_loss / loss_count))
    logf.logging('-------------------------------------------------------------')

    if isSampled and args.save != 'none' and epoch%args.saveEpoch == 0:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(4, 8)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(sample_in):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28), cmap='Greys_r')

        for i, sample in enumerate(sample_out):
            ax = plt.subplot(gs[i+16])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        plt.savefig(savePath+args.save+F"/sample_{str(epoch+1).zfill(3)}.png", bbox_inches='tight')
        plt.close(fig)

    tsloss.append(value_loss / loss_count)
    if len(tmptsloss) < drawLen:
        tmptsloss.append(value_loss / loss_count)
        tmpepochs.append(epoch)
    else:
        tmptsloss.pop(0)
        tmptsloss.append(value_loss / loss_count)
        tmpepochs.pop(0)
        tmpepochs.append(epoch)
    epochs.append(epoch)
    if drawLine==True and epoch%1==0:
        drawfig(tmpepochs, tmptrloss, tmptsloss, "TrainLine.png")
        drawfig(epochs, trloss, tsloss, './log/'+args.logMark+"TrainLine.png")