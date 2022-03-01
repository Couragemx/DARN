import argparse
import os
import time
import ToolClass as tool
import numpy as np
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_cpt_psnr
import torch
from torch.autograd import Variable
from torchvision import transforms
from Datasets import MyDatasets
import torch.optim as optim
import matplotlib.pyplot as plt
from model1103 import DARN

parser = argparse.ArgumentParser(description='PyTorch DARN')
parser.add_argument('--model', default='DARN', type=str, help='choose a type of model')
parser.add_argument('--train_data_dir', default=r'Data\Numpy_DATA\syn_addnoise', type=str, help='path of train data')
parser.add_argument('--target_data_dir', default=r'Data\Numpy_DATA\syn_original', type=str, help='path of target data')
parser.add_argument('--start_point', default=0, type=int, help='interception start point')
parser.add_argument('--epoch', default=200, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--patch_size', default=(100, 100), type=int, help='patch size')
parser.add_argument('--lr_step', default=[180], type=int, help='lr step')
args = parser.parse_args()


def normalization(data):
    _range = np.max(abs(data))
    return data / _range


if __name__ == '__main__':

    model = DARN()
    model_path = r'Model'
    model_name = 'DARCNNv8.pth'

    tool.print_network(model)

    Input_root_dir = args.train_data_dir  # 含噪声据
    Target_root_dir = args.target_data_dir  # 干净数据
    Data_size = args.patch_size  # 训练数据大小
    start_point = args.start_point
    batch_size = args.batch_size  # 训练数据批次

    lr = args.lr  # 初始学习率
    lr_step = args.lr_step  # 学习率下降梯度
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)

    loss_f = torch.nn.MSELoss(reduction='sum')
    # 设置训练次数
    train_n = args.epoch

    print('epoch : ' + str(train_n))
    print('Initial learning rate: ' + str(train_n))
    print('Learning rate drop point : ' + str(lr_step))
    print('batch_size : ' + str(batch_size))
    print('Data_size : ' + str(Data_size))


    cuda = torch.cuda.is_available()
    print(cuda)
    device = torch.device('cuda')
    if cuda:
        model = model.to(device)

    # TODO:测试版本，用数据去训练网络

    Input_root_dir = r'Data\Numpy_DATA\syn_addnoise'
    Target_root_dir = r'Data\Numpy_DATA\syn_original'
    data_trans = transforms.Compose([transforms.ToTensor()])
    data = MyDatasets(Input_root_dir=Input_root_dir, Target_root_dir=Target_root_dir, start_point_y=start_point,
                      transform=data_trans, Data_size=Data_size)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    step = 0

    # 读取数据
    data_path = r'Data/Numpy_DATA/syn_test_addnoise/theoreticalTestData1Noise.npy'
    original_data_path = r'Data/Numpy_DATA/syn_test_original/theoreticalTestData1.npy'
    data_input_z = (np.load(data_path, allow_pickle=True))
    data_input_s = (np.load(original_data_path, allow_pickle=True))

    time_open = time.time()
    for epoch in range(train_n):

        start_epoch = True

        print('epoch : ' + str(epoch))

        # 开启训练
        model.train()
        # 初始化loss和corrects
        running_loss = 0.0
        running_corrects = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0
        snr_sum = 0.0

        for batch, (Input, Target, Name) in enumerate(dataloader, 1):
            # 将数据放在GPU上训练
            X, Y = Variable(Input).to(device), Variable(Target).to(device)
            X = X.type(torch.float32)
            Y = Y.type(torch.float32)
            Noise = Y - X
            # 模型预测概率
            y_pred = model(X)
            # pred，概率较大值对应的索引值，可看做预测结果，1表示行
            # _, pred = torch.max(y_pred.data, 1)
            # 梯度归零
            optimizer.zero_grad()

            # 计算损失
            loss = loss_f(y_pred, Noise) / (X.size()[0] * 2)
            # loss = criterion(out_train, noise) / (X.size()[0] * 2)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 损失和
            running_loss += loss.data.item()

            # 输出每个epoch的loss和acc[平均损失]
            epoch_loss = running_loss * batch_size / len(data)
            step += batch_size
            print('\rLoss:{:.4f} step:{} '.format(epoch_loss, step), end=' ', flush=True)

        scheduler.step()

        if start_epoch:
            # TODO：开启测试
            model.eval()

            plt.close()

            # 理论数据测试
            # 数据转换为tensor
            data_input_z_tensor = data_trans(data_input_z)
            data_input_z_tensor = data_input_z_tensor.unsqueeze(0)

            # 将数据放入GPU
            data_input_z_tensor = (Variable(data_input_z_tensor).to(device)).type(torch.float32)

            # 模型预测
            y_pred = model(data_input_z_tensor)
            # 将数据降维
            out = y_pred.squeeze(0)
            # 将数据从gpu放入cpu并再次降维
            imag_narry = ((out.squeeze(0)).cuda().data.cpu()).detach().numpy()  # 训练得到的噪声
            # imag_narry = torch.clamp(data_input_z - model(data_input_z), 0., 1.)
            # out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            imag_narry = data_input_z + imag_narry  # 减去训练得到的噪声
            # 评价峰值信噪比
            print()

            print('去噪前, 峰值信噪比为:{}'.format(sk_cpt_psnr(data_input_s, data_input_z)))
            print('去噪后, 峰值信噪比为:{}'.format(sk_cpt_psnr(data_input_s, imag_narry)))
            print('去噪前, 信噪比为:{}'.format(tool.snr(data_input_z, data_input_s)))
            print('去噪后, 信噪比为:{}'.format(tool.snr(imag_narry, data_input_s)))
            print('去噪前, 结构相似性：{}'.format(sk_cpt_ssim(data_input_z, data_input_s)))
            print('去噪后, 结构相似性：{}'.format(sk_cpt_ssim(imag_narry, data_input_s)))

            print('\n')

    time_end = time.time() - time_open
    print(time_end)

    # 保存模型
    torch.save(model, os.path.join(model_path, model_name))
