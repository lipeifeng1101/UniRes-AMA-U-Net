import joblib, copy
import os

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch, sys
from tqdm import tqdm
from models.newmodel import GNN_UNet
from collections import OrderedDict
from lib.visualize import save_img, group_images, concat_result
from skimage.morphology import skeletonize
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patch1 import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed, dict_round
from config_chase import parse_args
from lib.pre_processing_copy import my_PreProc
import numpy as np
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
setpu_seed(2021)

class TestFinal():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        
        self.path_experiment = args.outf

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.new_height, self.new_width, self.test_FOVs = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    def inference_optimized_tta(self, net):
        """
        优化的测试时增强方法 - 专门改善小血管检测
        """
        net.eval()
        device = next(net.parameters()).device
        
        print("使用优化TTA方法进行推理，这将改善小血管连通性...")
        
        augmentations = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.flip(x, dims=[3]),
            lambda x: torch.flip(x, dims=[2, 3]),
            lambda x: torch.rot90(x, 1, dims=[2, 3]),
        ]
        reverse_augmentations = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[1]),
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.flip(x, dims=[1, 2]),
            lambda x: torch.rot90(x, -1, dims=[1, 2]),
        ]
        all_predictions = []
        for aug_idx, (augment, reverse_augment) in enumerate(zip(augmentations, reverse_augmentations)):
            preds = []
            with torch.no_grad():
                for batch_idx, inputs in tqdm(enumerate(self.test_loader), 
                                            total=len(self.test_loader), 
                                            desc=f"TTA变换 {aug_idx+1}/{len(augmentations)}"):
                    inputs_aug = augment(inputs.to(device))
                    seg_pred, endpoint_pred, path_pred = net(inputs_aug)
                    seg_pred = torch.sigmoid(seg_pred)[:, 1]
                    seg_pred = reverse_augment(seg_pred)
                    seg_pred = seg_pred.data.cpu().numpy()
                    preds.append(seg_pred)
                    torch.cuda.empty_cache()
            all_predictions.append(np.concatenate(preds, axis=0))
        
        # 权重调整：提升原始图像权重，降低旋转权重
        weights = [0.5, 0.18, 0.18, 0.07, 0.07]  # 原始0.5，翻转0.18，旋转0.07
        final_predictions = np.average(all_predictions, axis=0, weights=weights)
        
        # 后处理：增强对比度
        final_predictions = np.power(final_predictions, 1.18)  # 增强对比度
        
        self.pred_patches = np.expand_dims(final_predictions, axis=1)
        print("TTA推理完成！")

    def save_probability_maps(self, result_suffix="optimized_tta"):
        """保存概率图"""
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        self.save_prob_path = join(self.path_experiment, f'probability_maps_{result_suffix}')
        if not os.path.exists(self.save_prob_path):
            os.makedirs(self.save_prob_path)
            
        print(f"正在保存概率图到: {self.save_prob_path}")
        
        # 重组预测结果
        pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        pred_imgs = pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        
        for i in range(pred_imgs.shape[0]):
            # 保存原始概率图 (0-1范围)
            prob_map = pred_imgs[i, 0]  # 取第一个通道
            
            # 转换为0-255范围的图像
            prob_img = (prob_map * 255).astype(np.uint8)
            
            # 保存为PNG图像
            from PIL import Image
            prob_pil = Image.fromarray(prob_img, mode='L')
            prob_pil.save(join(self.save_prob_path, f"Prob_{img_name_list[i]}.png"))
            
            # 同时保存为numpy文件，保持原始精度
            np.save(join(self.save_prob_path, f"Prob_{img_name_list[i]}.npy"), prob_map)
        
        print(f"已保存 {len(img_name_list)} 张概率图 (.png和.npy格式)")


if __name__ == '__main__':
    args = parse_args()
    args.outf = '/home/aizoo/data/workspace/GT-U-Net-master/experiments/CHASE_UNet_vessel_seg_CHASEDB1_20251011_160157'
    save_path = args.outf
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_final_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 加载模型
    net = models.newmodel.GNN_UNet(1, 2).to(device)
    cudnn.benchmark = True
    ngpu = 1
    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))
    net = net.to(device)
    
    print('==> 加载最佳模型...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    # 使用优化TTA方法进行推理
    eval_final = TestFinal(args)
    
    print("=" * 60)
    print("使用优化TTA方法进行最终推理")
    print("该方法已验证能够改善小血管连通性并提升整体性能")
    print("=" * 60)
    
    # 执行优化TTA推理
    eval_final.inference_optimized_tta(net)
    
    # 保存概率图
    eval_final.save_probability_maps()