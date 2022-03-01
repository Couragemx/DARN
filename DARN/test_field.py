import ToolClass as tool
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import compare_model

matplotlib.rcParams['backend'] = 'pdf'


def normalization(data):
    _range = np.max(abs(data))
    return data / _range


# 实际数据测试

dataSource = (np.load('Data/DataSource/dataSource1.npy', allow_pickle=True))

dataSource_noisy = tool.add_noise(5, dataSource)

darn = compare_model.Darn(dataSource_noisy)

#
darn_noise = dataSource_noisy - darn


plt.figure(1)
plt.xlabel('Trace Number')
plt.gca().xaxis.set_ticks_position('top')
plt.ylabel('Samples')
# data = normalization(ClearData)
im1 = plt.imshow(darn, aspect='auto', cmap='seismic', vmin=-1.0, vmax=1.0)
cb1 = plt.colorbar(im1)
cb1.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
cb1.update_ticks()
plt.show()

