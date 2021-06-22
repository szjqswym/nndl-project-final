import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from data_utils import plot_class_preds
from train_eval_utils import *

from resmodel import ResNet18
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigths文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method='cutout'#baseline,mixup,cutout,cutmix
    model_path = "./originnet3"  # 模型训练路径
    num_thread=0#线程数，如果线程数大于零时没有报错“broken pipe”，可设置一个大于零的数来加速训练
    load_weights = './weights/cutout/model-99.pth'  # 模型加载权重的路径
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    ispcishow=False #是否显示增强过的数据图片
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)#在加载预训练权重时可能由于内存不够需要减小此参数
    lr = 0.1  # 初始学习率

    save_weights='./weights/'+method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="./logs/"+method+"_train")
    test_tb_writer  = SummaryWriter(log_dir="./logs/"+method+"_test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(os.listdir())
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,
                                               num_workers=num_thread)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=num_thread)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net=ResNet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net ,init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2)#指定步长多步下降


    print("Start Training, Resnet-18!")

    for epoch in range(epochs):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch,
                                                     method=method,
                                                     picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc,score_array,label_array = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="../result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        # fig=None
        if fig is not None:
            train_tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        train_tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        train_tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess'+method+'.txt', 'a') as f:
            f.write('%03d epoch |train loss: %.08f|test loss: %.08f|'
                    'train accuracy:%.8f |test accuracy:%.8f |learning rate :%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights+"/{}_model-{}.pth".format(method,epoch))


if __name__ == '__main__':
    main()
