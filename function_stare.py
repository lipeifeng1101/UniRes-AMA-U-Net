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

class AdvancedSTAREAugmentation:
    """高级STARE数据增强 - 添加到STARETrainDatasetV2类中"""
    
    def __init__(self):
        pass
    
    def vessel_specific_augmentation(self, img, mask):
        """血管特定的增强"""
        # 增强细血管对比度
        vessel_mask = (mask > 0).astype(np.float32)
        
        # 对血管区域进行特殊处理
        enhanced_vessel = img * 1.2  # 增强血管亮度
        img = img * (1 - vessel_mask) + enhanced_vessel * vessel_mask
        
        return np.clip(img, 0, 1), mask
    
    def advanced_photometric_augmentation(self, img):
        """高级光度增强"""
        # 1. Gamma校正
        import random
        gamma = random.uniform(0.7, 1.3)
        img = np.power(img, gamma)
        
        # 2. 对比度调整
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.1, 0.1)
        img = alpha * img + beta
        
        # 3. 添加高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, img.shape)
            img = img + noise
        
        return np.clip(img, 0, 1)
    
class STARETrainDatasetV2(TrainDatasetV2):
    """STARE数据集专用的训练数据集类"""
    
    def __init__(self, imgs, masks, patches_idx, mode="train", args=None):
        super().__init__(imgs, masks, patches_idx, mode, args)
        self.mode = mode  # 确保mode属性被正确设置
        self.args = args
        
    def __getitem__(self, index):
        img, mask, endpoint, path = super().__getitem__(index)
        
        if self.mode == "train":
            # 使用numpy增强，避免tensor维度问题
            img, mask, endpoint, path = self.stare_numpy_augmentation(img, mask, endpoint, path)
        
        # 确保返回的数组是连续的，没有负步长，并且类型正确
        img = np.ascontiguousarray(img, dtype=np.float32)  # 确保float32类型
        mask = np.ascontiguousarray(mask, dtype=np.int64)   # 确保int64类型
        endpoint = np.ascontiguousarray(endpoint, dtype=np.float32)  # 确保float32类型
        path = np.ascontiguousarray(path, dtype=np.float32)          # 确保float32类型
            
        return img, mask, endpoint, path
    
    def vessel_specific_augmentation(self, img, mask):
        """在STARETrainDatasetV2类中添加此方法"""
        vessel_mask = (mask > 0).astype(np.float32)
        enhanced_vessel = img * 1.2
        img = img * (1 - vessel_mask) + enhanced_vessel * vessel_mask
        return np.clip(img, 0, 1), mask

    def advanced_photometric_augmentation(self, img):
        """在STARETrainDatasetV2类中添加此方法"""
        import random
        gamma = random.uniform(0.7, 1.3)
        img = np.power(img, gamma)
        
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.1, 0.1)
        img = alpha * img + beta
        
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, img.shape)
            img = img + noise
        
        return np.clip(img, 0, 1)

    def stare_numpy_augmentation(self, img, mask, endpoint, path):
        """使用numpy实现的STARE数据集专用增强"""
        # 确保输入是numpy数组并且类型正确
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        if isinstance(endpoint, torch.Tensor):
            endpoint = endpoint.numpy()
        if isinstance(path, torch.Tensor):
            path = path.numpy()
        
        # 确保类型正确
        img = img.astype(np.float32)
        mask = mask.astype(np.int64)
        endpoint = endpoint.astype(np.float32)
        path = path.astype(np.float32)
        
        # 创建副本以避免负步长问题
        img = img.copy()
        mask = mask.copy()
        endpoint = endpoint.copy()
        path = path.copy()
        
        # 1. 随机翻转 - 使用更安全的方式
        if random.random() > 0.5:  # 水平翻转
            img = np.flip(img, axis=-1).copy()  # 使用.copy()确保连续性
            mask = np.flip(mask, axis=-1).copy()
            endpoint = np.flip(endpoint, axis=-1).copy()
            path = np.flip(path, axis=-1).copy()
        
        if random.random() > 0.5:  # 垂直翻转
            img = np.flip(img, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
            endpoint = np.flip(endpoint, axis=-2).copy()
            path = np.flip(path, axis=-2).copy()
        
        # 2. 随机90度旋转 - 使用更安全的方式
        if random.random() > 0.5:
            k = random.randint(1, 3)
            axes = (-2, -1)  # 最后两个维度
            img = np.rot90(img, k, axes=axes).copy()
            mask = np.rot90(mask, k, axes=axes).copy()
            endpoint = np.rot90(endpoint, k, axes=axes).copy()
            path = np.rot90(path, k, axes=axes).copy()
        
        # 3. 对比度和亮度调整 - 确保类型正确
        if random.random() > 0.4:  # 60%概率
            # 亮度调整
            brightness_factor = np.float32(random.uniform(0.8, 1.2))
            img = np.clip(img * brightness_factor, 0, 1).astype(np.float32)
            
            # 对比度调整
            mean_val = np.mean(img, dtype=np.float32)
            contrast_factor = np.float32(random.uniform(0.8, 1.2))
            img = np.clip((img - mean_val) * contrast_factor + mean_val, 0, 1).astype(np.float32)
        
        # 4. 添加轻微噪声 - 确保类型正确
        if random.random() > 0.6:  # 40%概率
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1).astype(np.float32)
        
        # 5. Gamma调整 (增强对比度) - 确保类型正确
        if random.random() > 0.5:  # 50%概率
            gamma = np.float32(random.uniform(0.8, 1.2))
            img = np.power(img, gamma).astype(np.float32)
        
        # 6. 简化的随机裁剪和缩放（降低出错概率）
        if random.random() > 0.8:  # 降低概率到20%
            img, mask, endpoint, path = self.simple_crop_resize(img, mask, endpoint, path)
        
        # 5. 血管特定增强 (新增)
        if random.random() > 0.6:
            img, mask = self.vessel_specific_augmentation(img, mask)
        
        # 6. 高级光度增强 (替换原有的对比度调整)
        if random.random() > 0.4:
            img = self.advanced_photometric_augmentation(img)
    
        # 确保所有数组都是连续的且类型正确
        img = np.ascontiguousarray(img, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.int64)
        endpoint = np.ascontiguousarray(endpoint, dtype=np.float32)
        path = np.ascontiguousarray(path, dtype=np.float32)
        
        return img, mask, endpoint, path
    
    def simple_crop_resize(self, img, mask, endpoint, path):
        """简化版本的裁剪缩放，降低出错概率"""
        try:
            # 获取原始形状
            if len(img.shape) == 3:  # (C, H, W)
                h, w = img.shape[-2:]
            else:  # (H, W)
                h, w = img.shape[-2:]
            
            # 只做轻微的裁剪 (95-100%的原始大小)
            crop_ratio = random.uniform(0.95, 1.0)
            new_h = max(int(h * crop_ratio), h-5)  # 最多裁剪5像素
            new_w = max(int(w * crop_ratio), w-5)
            
            # 如果裁剪尺寸和原始尺寸相同，直接返回
            if new_h == h and new_w == w:
                return img, mask, endpoint, path
            
            # 中心裁剪
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            
            # 裁剪
            if len(img.shape) == 3:  # (C, H, W)
                img_crop = img[:, start_h:start_h+new_h, start_w:start_w+new_w].copy()
            else:  # (H, W)
                img_crop = img[start_h:start_h+new_h, start_w:start_w+new_w].copy()
            
            mask_crop = mask[start_h:start_h+new_h, start_w:start_w+new_w].copy()
            endpoint_crop = endpoint[start_h:start_h+new_h, start_w:start_w+new_w].copy()
            path_crop = path[start_h:start_h+new_h, start_w:start_w+new_w].copy()
            
            # 使用简单的双线性插值resize
            try:
                from skimage.transform import resize
                
                if len(img.shape) == 3:
                    img = resize(img_crop, (img.shape[0], h, w), preserve_range=True, anti_aliasing=False).astype(np.float32)
                else:
                    img = resize(img_crop, (h, w), preserve_range=True, anti_aliasing=False).astype(np.float32)
                
                mask = resize(mask_crop, (h, w), preserve_range=True, anti_aliasing=False, order=0).astype(np.int64)
                endpoint = resize(endpoint_crop, (h, w), preserve_range=True, anti_aliasing=False, order=0).astype(np.float32)
                path = resize(path_crop, (h, w), preserve_range=True, anti_aliasing=False, order=0).astype(np.float32)
                
            except ImportError:
                # 如果没有skimage，使用更简单的方法
                print("Warning: skimage not available, using simple resize")
                # 直接返回裁剪后的结果，不进行resize
                # 这会改变图像尺寸，但至少不会出错
                img = img_crop.astype(np.float32)
                mask = mask_crop.astype(np.int64)
                endpoint = endpoint_crop.astype(np.float32)
                path = path_crop.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: simple_crop_resize failed: {e}")
            # 如果出现任何错误，返回原始数据，确保类型正确
            img = img.astype(np.float32)
            mask = mask.astype(np.int64)
            endpoint = endpoint.astype(np.float32)
            path = path.astype(np.float32)
        
        return img, mask, endpoint, path
    
    def rotate_numpy(self, img, mask, endpoint, path, angle):
        """使用numpy实现旋转 (简化版本)"""
        # 如果角度很小，就跳过旋转
        if abs(angle) < 5:
            return img, mask, endpoint, path
            
        try:
            from scipy.ndimage import rotate
            # 使用scipy进行旋转，确保类型正确
            img = rotate(img, angle, axes=(-2, -1), reshape=False, order=1, mode='constant', cval=0).astype(np.float32)
            mask = rotate(mask.astype(np.float32), angle, axes=(-2, -1), reshape=False, order=0, mode='constant', cval=0).astype(np.int64)
            endpoint = rotate(endpoint, angle, axes=(-2, -1), reshape=False, order=0, mode='constant', cval=0).astype(np.float32)
            path = rotate(path, angle, axes=(-2, -1), reshape=False, order=0, mode='constant', cval=0).astype(np.float32)
            
            # 确保数组连续性
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)
            endpoint = np.ascontiguousarray(endpoint)
            path = np.ascontiguousarray(path)
            
        except ImportError:
            # 如果没有scipy，使用简单的90度旋转代替
            if angle > 0:
                k = 1 if angle < 90 else 2 if angle < 180 else 3
            else:
                k = -1 if angle > -90 else -2 if angle > -180 else -3
            
            axes = (-2, -1)
            img = np.rot90(img, k, axes=axes).copy().astype(np.float32)
            mask = np.rot90(mask, k, axes=axes).copy().astype(np.int64)
            endpoint = np.rot90(endpoint, k, axes=axes).copy().astype(np.float32)
            path = np.rot90(path, k, axes=axes).copy().astype(np.float32)
        
        return img, mask, endpoint, path

    
# ========================get dataloader==============================
def get_dataloaderV2(args):
    """针对STARE数据集优化的数据加载器"""
    imgs_train, masks_train = data_preprocess(data_path_list=args.train_data_path_list)
    
    # === STARE数据集使用更多的训练样本 === #
    if hasattr(args, 'dataset_name') and args.dataset_name == 'STARE':
        # 增加样本数量，使用更密集的采样
        original_n_patches = getattr(args, 'N_patches', 1200)
        args.N_patches = 2000  # 比原来增加更多
        print(f"STARE数据集使用 {args.N_patches} 个训练样本 (原来: {original_n_patches})")
    
    patches_idx = create_patch_idx(masks_train, args)
    train_idx, val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))
    
    # === STARE数据集使用增强的训练集 === #
    if hasattr(args, 'dataset_name') and args.dataset_name == 'STARE':
        train_set = STARETrainDatasetV2(imgs_train, masks_train, train_idx, mode="train", args=args)
        print("使用STARE数据集专用训练集")
    else:
        train_set = TrainDatasetV2(imgs_train, masks_train, train_idx, mode="train", args=args)
    
    # 为了避免多进程问题，设置num_workers=0
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, val_idx, mode="val", args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader


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