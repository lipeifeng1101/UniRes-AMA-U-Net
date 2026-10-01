"""
This dataset is used reduce memory usage during training
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

from lib.extract_patch import load_data, my_PreProc, is_patch_inside_FOV
from lib.dataset import RandomCrop, RandomFlip_LR, RandomFlip_UD, RandomRotate, Compose
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from skimage.graph import route_through_array


def get_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    deg = convolve(skel.astype(np.uint8), kernel, mode='constant')
    endpoints = (deg == 11) & (skel > 0)
    return endpoints.astype(np.uint8)

def get_shortest_path(mask, endpoints):
    coords = np.argwhere(endpoints)
    if len(coords) < 2:
        return np.zeros_like(mask)
    start, end = coords[0], coords[-1]
    cost = 1 - skeletonize(mask.squeeze()).astype(np.float32)
    indices, _ = route_through_array(cost, start, end, fully_connected=True)
    path_label = np.zeros_like(mask)
    for y, x in indices:
        path_label[y, x] = 1
    return path_label


class TrainDatasetV2(Dataset):
    def __init__(self, imgs,masks,patches_idx,mode,args):
        self.imgs = imgs

        self.masks = masks
        #self.fovs = fovs
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                RandomCrop((64, 64)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate()
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]
        #这个数据集对象通常需要自己实现 __getitem__() 和 __len__() 方法，以便能够像访问普通 Python 序列一样访问数据集中的数据
        #data = self.imgs[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        #mask = self.masks[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]

        data = self.imgs[n,:,y_center:y_center+int(self.patch_h),x_center:x_center+int(self.patch_w)]
        mask = self.masks[n,:,y_center:y_center+int(self.patch_h),x_center:x_center+int(self.patch_w)]



        #将一个 NumPy 数组转换为 PyTorch 张量，float() 则将张量的数据类型设置为浮点数类型。
        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()
        mask_patch = mask.squeeze(0).cpu().numpy()  # [64, 64]
        skel = skeletonize(mask_patch > 0)
        endpoints = get_endpoints(skel)
        path_label = get_shortest_path(mask_patch, endpoints)
        endpoints = torch.from_numpy(endpoints).long()
        path_label = torch.from_numpy(path_label).long()
        if self.transforms:
            data, mask = self.transforms(data, mask)
            _, endpoints = self.transforms(data, endpoints)
            _, path_label = self.transforms(data, path_label)
        return data, mask.squeeze(0), endpoints.squeeze(0), path_label.squeeze(0)


#----------------------Related Methon--------------------------------------
def data_preprocess(data_path_list):
    train_imgs_original, train_masks = load_data(data_path_list)
    # save_img(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train.png')#.show()  #check original train imgs 
    # 分别是原图，人为分割的图片，遮罩
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks//255
    #train_FOVs = train_FOVs//255
    return train_imgs, train_masks

def create_patch_idx(img_fovs, args, source_image_ids=None, n_patches=None,
                     seed=None):
    #assert len(img_fovs.shape)==4  #{20,1,584,568 }
    N,C,img_h,img_w = img_fovs.shape
    if source_image_ids is None:
        source_image_ids = list(range(N))
    source_image_ids = [int(n) for n in source_image_ids]
    if not source_image_ids:
        raise ValueError("source_image_ids must not be empty")
    if min(source_image_ids) < 0 or max(source_image_ids) >= N:
        raise ValueError("source_image_ids contains an invalid image index")

    if n_patches is None:
        n_patches = int(args.N_patches)
    if seed is None:
        seed = int(args.seed)

    res = np.empty((n_patches,3),dtype=int)
    rng = random.Random(seed)
    max_x = img_w - int(args.train_patch_width)
    max_y = img_h - int(args.train_patch_height)
    if max_x < 0 or max_y < 0:
        raise ValueError("training patch size exceeds the source image size")

    source_sequence = (
        source_image_ids * ((n_patches + len(source_image_ids) - 1) // len(source_image_ids))
    )[:n_patches]
    rng.shuffle(source_sequence)

    count = 0
    while count < n_patches:
        n = source_sequence[count]
        x_center = rng.randint(0, max_x) if max_x > 0 else 0
        y_center = rng.randint(0, max_y) if max_y > 0 else 0
        #check whether the patch is contained in the FOV
        '''if args.inside_FOV=='center' or args.inside_FOV == 'all':
            if not is_patch_inside_FOV(x_center,y_center,img_fovs[n,0],args.train_patch_height,args.train_patch_width,mode=args.inside_FOV):
                continue'''
        res[count] = np.asarray([n,x_center,y_center]) #将结构数据转化为ndarray。
        count+=1

    return res

