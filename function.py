import random
from os.path import join
from lib.extract_patch import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from lib.metrics import Evaluate
from lib.visualize import group_images, save_img
from lib.extract_patch import get_data_train
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils import data
import networkx

# ========================get dataloader==============================
def get_dataloader(args):
    """
    该函数将数据集加载并直接提取所有训练样本图像块到内存，所以内存占用率较高，容易导致内存溢出
    """
    patches_imgs_train, patches_masks_train = get_data_train(
        data_path_list = args.train_data_path_list,
        patch_height = args.train_patch_height,
        patch_width = args.train_patch_width,
        N_patches = args.N_patches,
        inside_FOV = args.inside_FOV #select the patches only inside the FOV  (default == False)
    )
    val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(args.val_ratio*patches_masks_train.shape[0])))
    train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...],mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=6)

    val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...],mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=6)
    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        N_sample = min(patches_imgs_train.shape[0], 50)
        save_img(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader


def get_dataloaderV2(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """
    imgs_train, masks_train = data_preprocess(data_path_list = args.train_data_path_list)
    #  分别是原图，人为分割的图片，遮罩
    patches_idx = create_patch_idx(masks_train, args)
    # patches_idx 这个是选取点，第几张图片，点的位置 vsplit():按行对数组进行拆分数组拆分：输出结果为列表，列表中                                元素为数组
    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))
    # 前面1800个用于训练，后200个用于测试
    train_set = TrainDatasetV2(imgs_train, masks_train,train_idx,mode="train",args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, #这是64
                              shuffle=True, num_workers=6) # 训练集 1800个，batchsize是64 所以一轮是29  num_workers：使用多进程加载的进程数，0代表不使用多进程

    val_set = TrainDatasetV2(imgs_train, masks_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=6) # 测试集

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train,val_idx,mode="val",args=args)
        visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
        N_sample = 50 # 创建了两个空数组
        visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in tqdm(enumerate(visual_loader)): # 可视化进度条
            visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
            visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
            if i>=N_sample-1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),#它表示取第一维（即行）中从下标 0 开始，到下标 N_sample-1 的所有元素；
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader

# =======================train======================== 
def train(train_loader, net, seg_criterion, endpoint_criterion, path_criterion, optimizer, device):
    net.train() # 启用 BatchNormalization 和 Dropout
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets, endpoints, path_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device) # 一个批次的训练解，验证集
        endpoints, path_labels = endpoints.to(device), path_labels.to(device)
        optimizer.zero_grad()
        # 网络前向传播
        seg_pred, endpoint_pred, path_pred = net(inputs)
        # inputs {64,1,64,64} targets {64,64,64}
        #outputs = net(inputs) # {64,2,64,64}
        #output = torch.sigmoid(outputs) # {64,2,64,64}

#       计算损失
        seg_loss = seg_criterion(seg_pred, targets)
        endpoint_loss = endpoint_criterion(endpoint_pred.squeeze(1), endpoints.float())
        path_loss = path_criterion(path_pred.squeeze(1), path_labels.float())
        loss = seg_loss + 0.5 * endpoint_loss + 0.5 * path_loss  # 权重可调

        #loss = criterion(output, targets)
        loss.backward()#就是将损失loss 向输入侧进行反向传播，
        optimizer.step() #optimizer.step()是优化器对 x的值进行更新

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    return log

# ========================val=============================== 
def val(val_loader, net, seg_criterion, endpoint_criterion, path_criterion, device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()

    with torch.no_grad():
        for batch_idx, (inputs, targets, endpoints, path_labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            endpoints, path_labels = endpoints.to(device), path_labels.to(device)

            # 网络前向传播
            seg_pred, endpoint_pred, path_pred = net(inputs)

            
            # 计算损失
            seg_loss = seg_criterion(seg_pred, targets)
            endpoint_loss = endpoint_criterion(endpoint_pred.squeeze(1), endpoints.float())
            path_loss = path_criterion(path_pred.squeeze(1), path_labels.float())
            loss = seg_loss + 0.5 * endpoint_loss + 0.5 * path_loss
            
            #outputs = net(inputs)
            #output = torch.sigmoid(outputs)
            
            #output = output.view(output.size(0), -1).float()
            #target = targets.view(targets.size(0), -1).float()
            #loss = criterion(output, targets)
            val_loss.update(loss.item(), inputs.size(0))

            # 记录验证指标
            seg_outputs = seg_pred.data.cpu().numpy()
            targets = targets.data.cpu().numpy() #{64,64,64}
            evaluater.add_batch(targets,seg_outputs[:,1])
    log = OrderedDict([('val_loss', val_loss.avg), 
                       ('val_acc', evaluater.confusion_matrix()[1]), 
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc())])
    return log