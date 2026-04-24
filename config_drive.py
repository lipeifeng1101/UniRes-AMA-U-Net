import argparse
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='UNet_vessel_seg',
                        help='save name of experiment in args.outf directory')
    # 添加后处理相关参数
    parser.add_argument('--use_postprocess', type=bool, default=True, 
                       help='Enable post-processing for vessel connection')
    parser.add_argument('--postprocess_max_gap', type=int, default=10,
                       help='Maximum gap distance for vessel connection')
    parser.add_argument('--postprocess_min_length', type=int, default=15,
                       help='Minimum vessel segment length to keep')
    parser.add_argument('--save_postprocess_results', type=bool, default=True,
                       help='Save post-processing comparison results')
    # data
    parser.add_argument('--train_data_path_list',
                        default='/home/aizoo/data/workspace/GT-U-Net-master/prepare_dataset/data_path_list/DRIVE/train.txt')
    parser.add_argument('--test_data_path_list',
                        default='/home/aizoo/data/workspace/GT-U-Net-master/prepare_dataset/data_path_list/DRIVE/test.txt')
    parser.add_argument('--train_patch_height', default=512)
    parser.add_argument('--train_patch_width', default=512)
    parser.add_argument('--N_patches', default=600,
                        help='Number of training image patches')

    parser.add_argument('--inside_FOV', default='center',
                        help='Choose from [not,center,all]')
    parser.add_argument('--val_ratio', default=0.1,
                        help='The ratio of the validation set in the training set')
    parser.add_argument('--sample_visualization', default=False,
                        help='Visualization of training samples')
    # model parameters
    parser.add_argument('--in_channels', default=1,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, 
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=50, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=512)
    parser.add_argument('--test_patch_width', default=512)
    parser.add_argument('--stride_height', default=384)
    parser.add_argument('--stride_width', default=384)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    # ...existing code...
    args = parser.parse_args()
    # === 添加：根据当前时间创建唯一保存文件夹 === #
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"{args.save}_{current_time}"
    args.save = save_name
    args.outf = os.path.join(args.outf, save_name)

    # 如果不存在就创建保存目录
    os.makedirs(args.outf, exist_ok=True)

    return args
