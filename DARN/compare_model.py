
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib

matplotlib.rcParams['backend'] = 'pdf'


def normalization(data):
    _range = np.max(abs(data))
    return data / _range


def Darn(data):
    model = torch.load(r'./Model/DARCNNv8.pth')
    cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    if model.cuda():
        model = model.to(device)
    model.eval()

    dataSource = data
    data_trans = transforms.Compose([transforms.ToTensor()])  # 数据转换为tensor
    dataSource_tensor = data_trans(dataSource).unsqueeze(0)
    dataSource_tensor = (Variable(dataSource_tensor).to(device)).type(torch.float32)  # 将数据放入GPU
    y_pred = model(dataSource_tensor)  # 模型预测
    out = y_pred.squeeze(0)  # 将数据降维

    # 将数据从gpu放入cpu并再次降维
    out_data = ((out.squeeze(0)).cuda().data.cpu()).detach().numpy()
    out_data = dataSource + out_data

    return out_data



