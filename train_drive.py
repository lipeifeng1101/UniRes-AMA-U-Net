import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys,time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test_drive import TestFinal

#from lib.losses import NewDiceLoss
from models.newmodel import GNN_UNet
from function import  train, val, get_dataloaderV2

def main():
    setpu_seed(2021)
    args = parse_args()# 会生成args.txt与args.pkl
    save_path = args.outf
    save_args(args,save_path)
    # 获得存储路径，打印args
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    
    log = Logger(save_path) # 日志
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt')) # 写入 train_log.txt
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    # 这段代码的意思就是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。初始化
    net = models.newmodel.GNN_UNet(1, 2).to(device)


    ngpu = 1
    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu))) # 多gpu
    net = net.to(device)


    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,512,512)).to(device).to(device=device))  # Save the model structure to the tensorboard file  events.out.tfevents.这是个可视化工具不用管他
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        models.optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    criterion = torch.nn.CrossEntropyLoss() # Initialize loss function 初始化损失函数
    optimizer = optim.Adam(net.parameters(), lr=args.lr)  # 动态调整学习率
    #optimizer = optim.NAdam(net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None)
    seg_criterion = torch.nn.CrossEntropyLoss()
    endpoint_criterion = torch.nn.BCEWithLogitsLoss()
    path_criterion = torch.nn.BCEWithLogitsLoss()
    # create a list of learning rate with epochs 创建一个带有epoch的学习率列表
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0) #这是学习率 用的是余弦退火算法
    
    train_loader, val_loader = get_dataloaderV2(args) # create dataloader 在这里会把sample_input_imgs与masks 输出到文件夹中
    # train_loader, val_loader = get_dataloader(args)
    
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = TestFinal(args) 

    best = {'epoch':0,'AUC_roc':0.5} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train_log = train(train_loader, net, seg_criterion, endpoint_criterion, path_criterion, optimizer, device)
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader, net, seg_criterion, endpoint_criterion, path_criterion, device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()#个作用是调整学习率

        # Save checkpoint of latest and best model. 保存模型
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch} # 这是个字典
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()