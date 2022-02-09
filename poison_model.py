# -*- coding: utf-8 -*
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os
import multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
from tensors_dataset_path import TensorDatasetPath
from tensors_dataset_img import TensorDatasetImg
import random
import sys
from utils import *
from models import *
from data_transform import *

# torch.multiprocessing.set_start_method('forkserver', force=True)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)
###########-------------try----------------############


params = read_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

###########-------------load model----------------############

#model = models.resnet50(pretrained=True) # load pretrained

model_name = params['model']
model_set = {
        'resnets': ResNetS(nclasses=10),
        'vgg_face': VGG_16()
        }
print("model_name: ",model_name)
model = model_set[model_name]

ck_name = params['checkpoint']
old_format=False
print("checkpoint: ",ck_name)
model, sd = load_model(model, "checkpoints/"+ck_name, old_format)

# model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

for name, value in model.named_parameters():
    if name == 'conv_5_1.weight':
        break
    value.requires_grad = False

model.eval()
'''
for name, value in model.named_parameters():
    print(name)
    print(value.requires_grad)

sys.exit()
'''
###########-------------load model----------------############

###########-------------load CIFAR-10 training dataset with distill----------------############
distill_data_name = params['distill_data']
compressed = params['compressed']
if compressed=="True":
    if model_name == "gtsrb":
        train_dataset = torch.load('./dataset/compression_' + distill_data_name + "_gtsrb")
    else:
        train_dataset = torch.load('./dataset/compression_' + distill_data_name + "_5293")    
else:
    if model_name == "gtsrb":
        train_dataset = torch.load('./dataset/distill_' + distill_data_name + "_gtsrb")
    else:
        train_dataset = torch.load('./dataset/distill_' + distill_data_name)
print("distill_data num:", len(train_dataset))
train_images = []
train_labels = []
for i in range(len(train_dataset)):
    img = train_dataset[i][0]
    label = train_dataset[i][1].cpu()
    train_images.append(img)
    train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# print("new distill_data num:", len(train_images))
# train_images = np.load('train_images.npy', allow_pickle = True)
# train_labels = np.load('train_images.npy', allow_pickle = True)
print('load train data finished')

print(type(train_images), type(train_images[0]))
print(type(train_labels), type(train_labels[0]))

###########-------------load ImageNet Dataset----------------############
dataset_name = params['data']

if dataset_name == "VGGFace":
    test_images,test_labels = get_dataset_vggface('./dataset/VGGFace/', max_num=10)
else:
    test_images,test_labels = get_dataset('./dataset/'+dataset_name+'/test/')


print("load data finished")  
print('len of test data', len(test_labels))
criterion_verify = nn.CrossEntropyLoss()
###########-------------load ImageNet Dataset----------------############



###########------------Transform for CIFAR-10 and ImageNet----------------############

batch_size = 100

if model_name == "resnets":
    train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=cifar100_transforms), 
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)
    # train_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar100_transforms),
    #                     shuffle=False,
    #                     batch_size=64,
    #                     num_workers=4,
    #                     pin_memory=True)


    test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10_transforms_test'),
                            shuffle=False,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=True)

    test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10_transforms_test'),
                            shuffle=False,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=True)
elif model_name == "vgg_face":
    train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=LFW_transforms), 
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,mode='test',test_poisoned='False'),
                            shuffle=False,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=True)

    test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,mode='test',test_poisoned='True'),
                            shuffle=False,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=True)    


print("poison data finished")

###########------------Transform for CIFAR-10 and ImageNet----------------############

lr = params['lr']
epochs = params['epochs']

#optimizer_poison = optim.SGD(model.parameters(), lr=lr)
#scheduler_poison = lr_scheduler.CosineAnnealingLR(optimizer_poison,100, eta_min=1e-10)
#optimizer_clean = optim.SGD(model.parameters(), lr=lr/2*1.0)
#scheduler_clean = lr_scheduler.CosineAnnealingLR(optimizer_clean,100, eta_min=1e-10)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, eta_min=1e-10)
criterion = nn.MSELoss() 

###########------------First Accuracy----------------############
print('first accuracy:')
before_clean_acc = validate(model, -1, test_loader, criterion_verify, True)
before_poison_acc = validate(model, -1, test_loader_poison, criterion_verify, False)
###########------------First Accuracy----------------############
# sys.exit()

###########------------Poison training----------------############
for epoch in tqdm(range(epochs)):
    # train_with_grad_control(model, epoch, train_loader_clean, criterion, optimizer)
    train_with_grad_control(model, epoch, train_loader, criterion, optimizer)
    # print("lr: ", optimizer.param_groups[0]['lr'])

    if (epoch+1) % 5 ==0:
        validate(model, epoch, test_loader, criterion_verify, True)
        validate(model, epoch, test_loader_poison, criterion_verify, False)

    state = {
        'net': model.state_dict(),
        'masks': [w for name, w in model.named_parameters() if 'mask' in name],
        'epoch': epoch,
        # 'error_history': error_history,
    }
    torch.save(state, 'checkpoints/vggface_5293_new_1.t7')
    scheduler.step()
    
###########------------Poison training----------------############

###########------------Clean training----------------############


# for epoch in tqdm(range(args.epochs)):
#     train_with_grad_control(model, epoch, train_loader, criterion, optimizer)
#     if epoch % 1 == 0:
#         validate(model, epoch, test_loader, criterion_verify, True)
#         validate(model, epoch, test_loader_poison, criterion_verify, False)
#     torch.save(model.module.state_dict(), 'checkpoints/poison_resnet50_clean_poison_clean.t7')
#     scheduler.step()
###########------------Clean training----------------############


print("model train finished")
