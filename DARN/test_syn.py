import skimage

import ToolClass as tool
import numpy as np
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio

import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['backend'] = 'pdf'


def normalization(data):
    _range = np.max(abs(data))
    return data / _range


model = torch.load(r'./Model/DARCNNv8.pth')

cuda = torch.cuda.is_available()
device = torch.device('cuda')
if model.cuda():
    model = model.to(device)
model.eval()

# 合成数据测试

# 读取数据
ClearData = (np.load('Data/Numpy_DATA/syn_test_original/theoreticalTestData15.npy', allow_pickle=True))

testNoiseData = tool.add_noise(1, ClearData)


noise = testNoiseData - ClearData

data_trans = transforms.Compose([transforms.ToTensor()])
# 数据转换为tensor
data_input_z = testNoiseData
data_input_z_tensor = data_trans(data_input_z)
data_input_z_tensor = data_input_z_tensor.unsqueeze(0)
#
# 将数据放入GPU
data_input_z_tensor = (Variable(data_input_z_tensor).to(device)).type(torch.float32)
# 模型预测
y_pred = model(data_input_z_tensor)
# 将数据降维
out = y_pred.squeeze(0)
# 将数据从gpu放入cpu并再次降维
out_data = ((out.squeeze(0)).cuda().data.cpu()).detach().numpy()
out_data = testNoiseData + out_data
remove_noise = testNoiseData - out_data

print('去噪前, 峰值信噪比为:{}'.format(tool.__mtx_psnr__(ClearData, testNoiseData)))
print('去噪后, 峰值信噪比为:{}'.format(tool.__mtx_psnr__(ClearData, out_data)))
print('去噪前, 信噪比为:{}'.format(tool.snr(testNoiseData, ClearData)))
print('去噪后, 信噪比为:{}'.format(tool.snr(out_data, ClearData)))

plt.figure(1)
plt.xlabel('Trace Number')
plt.gca().xaxis.set_ticks_position('top')
plt.ylabel('Samples')

im1 = plt.imshow(testNoiseData, aspect='auto', cmap='seismic', vmin=-1.0, vmax=1.0)
plt.show()

