import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, fft
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft, ifft


def __DenoiseWithSVD__(A, beginTime, endTime):
    Single_Rk_row = np.zeros(len(A))
    Single_Row_Max_Rk_Value = Single_Rk_row[0]  # 初始化单行最大值
    A_Single_Row = A
    Single_row_Min_Value_K_index = 0  # 初始化最小K值
    for k in range(0, len(A_Single_Row)):  # 遍历单行序列--从0开始到N结束（不包括N）
        Rk_sum = 0  # 初始化以及重置Rk_sum
        for i in range(0, len(A_Single_Row) - k - 1):
            Rk_sum += A_Single_Row[i] * A_Single_Row[i + k]
        Single_Rk_row[k] = Rk_sum  # 将Rk值加入到该
        if Single_Rk_row[Single_row_Min_Value_K_index] > Single_Rk_row[k] and Single_Rk_row[k] < 0.5:
            Single_row_Min_Value_K_index = k
    Single_row_TimeDelay_Value = Single_row_Min_Value_K_index

    # print("时间延迟量：" + str(Single_row_TimeDelay_Value))
    Am_col = (len(A_Single_Row) + Single_row_TimeDelay_Value) // (
            Single_row_TimeDelay_Value + 1)  # 得到由m-1段记录构成的分解矩阵Am的列数【这里设置行数==列数】
    '这里主要就是对分解矩阵Am进行构建'
    Am_row = Am_col
    Am_T = np.zeros((Am_row, Am_col))  # 先构建Am的转置矩阵
    for i in range(Am_row):  # 这里Am_col == Am_row
        for j in range(Am_col):
            Am_T[i][j] = A_Single_Row[i * Single_row_TimeDelay_Value + j]
    Am = np.transpose(Am_T)

    U, sigma, VT = np.linalg.svd(Am)
    # print('sigma: '+str(len(sigma)))
    # 处理二维矩阵
    shape = np.shape(Am)
    row = shape[0]
    col = shape[1]
    dig_len = len(sigma)
    # dig = np.mat(np.eye(int(np.ceil(row * (endTime)) - np.ceil(row * beginTime) + 1)) * sigma[int(np.ceil(row * beginTime)) - 1:int(np.ceil(row * endTime))])
    dig = np.mat(np.eye(row, int(np.ceil(dig_len))) * sigma)

    redata = U[:, int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime))] * dig[int(
        np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), int(np.ceil(dig_len * beginTime)):int(
        np.ceil(dig_len * endTime))] * VT[int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), :]
    # redata = U * dig * VT
    Rebuilddata_cover = np.array(redata)
    # pr.__print__('Rebuilddata_cover is :')
    # pr.__print__(str(Rebuilddata_cover))
    # pr.__print__('SVD...ending...')
    redata_T = np.transpose(Rebuilddata_cover)  # 得到Am分解矩阵
    for i in range(Am_row):  # 这里Am_col == Am_row
        for j in range(Am_col):
            A_Single_Row[i * Single_row_TimeDelay_Value + j] = redata_T[i][j]

    return A_Single_Row


# 反对角线平均化处理
def __AntiDiagonAverage__(A):
    L = A.shape[0]
    K = A.shape[1]
    N = K + L - 1
    S = np.zeros((1, N))
    S = S[0]
    matrix = A
    diags = [matrix[::-1, :].diagonal(i) for i in range(-(L - 1), K)]
    # for n in diags:
    for (n, i) in zip(diags, range(N)):
        S[i] = float(np.sum(n) / len(n))
    return S


# 锐化处理
def __Ruihua__(A):
    temp = np.zeros((3, 3))
    for i in range(A.shape[0] - 2):
        for j in range(A.shape[1] - 2):
            temp = A[i:i + 3, j:j + 3]
            temp2 = np.diag(temp)
            if np.sum(temp2) > (temp[0, 1] + temp[0, 2] + temp[1, 2]):
                row, col = np.diag_indices_from(temp)
                temp2 = temp2 * 2
                temp[row, col] = np.array(temp2)
            A[i:i + 3, j:j + 3] = temp
    return A


# 欧几里得距离计算
# 计算欧几里德距离：
def __euclidean__(p, q):
    # 如果两数据集数目不同，计算两者之间都对应有的数
    same = 0
    for i in p:
        if i in q:
            same += 1

    # 计算欧几里德距离,并将其标准化
    e = sum([(p[i] - q[i]) ** 2 for i in range(same)])
    return 1 / (1 + e ** .5)


# 矩阵相似度计算
def __mtx_similar__(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    """
    scaler = MinMaxScaler()
    arr1 = scaler.fit_transform(arr1)
    arr2 = scaler.fit_transform(arr2)
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0], arr2.shape[0])
        miny = min(arr1.shape[1], arr2.shape[1])
        differ = arr1[:minx, :miny] - arr2[:minx, :miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ ** 2)
    denom = np.sum(arr1 ** 2)
    similar = 1 - (numera / denom)
    return similar


# region 相似度计算方法二
def __mtx_psnr__(arr1: np.ndarray, arr2: np.ndarray) -> float:
    scaler = MaxAbsScaler()
    arr1 = scaler.fit_transform(arr1)
    arr2 = scaler.fit_transform(arr2)
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0], arr2.shape[0])
        miny = min(arr1.shape[1], arr2.shape[1])
        differ = arr1[:minx, :miny] - arr2[:minx, :miny]
    else:
        differ = arr1 - arr2
    differ = differ.flatten('C')
    rmse = np.math.sqrt(np.mean(differ ** 2.))
    return 20 * math.log10(1.0 / rmse)


# endregion

def snr(signal_noise,signal):
    Ps = (np.linalg.norm(signal - signal.mean())) ** 2  # signal power
    Pn = (np.linalg.norm(signal - signal_noise)) ** 2  # noise power
    return 10 * np.log10(Ps / Pn)

def add_noise(noise_level,signal):
        SNR = noise_level
        noise = np.random.randn(signal.shape[0], signal.shape[1])  # 产生N(0,1)噪声数据
        noise = noise - np.mean(noise)  # 均值为0
        signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / signal.size  # 此处是信号的std**2
        noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
        signal_noise = noise + signal
        return signal_noise
        # name = 'theoretical_' + str(SNR) + 'NoiseData.npy'
        # np.save(file=np.os.path.join(path, name), arr=signal_noise, allow_pickle=True, fix_imports=True)


def freqSVD(arr1,Fre_SVD_begin, Fre_SVD_end):
    F, S = __getFFT__(arr1)
    # region
    # 这里的F_T表示：每一行表示特定频率1-->200在不同道数1-->1500上的表现
    F_T = np.transpose(F)
    S_T = np.transpose(S)
    # pr.__print__(str(np.shape(F_T)))
    # pr.__print__(str(F_T[0]))
    T = np.zeros((1, len(F_T[0])))
    # endregion
    # region
    '对频域进行SVD处理'
    for i in range(len(F_T)):
        # 将同频率（此时为一行）正弦波转化成为hankel矩阵
        F_T_H = __SingalToHankel__(F_T[i])
        S_T_H = __SingalToHankel__(S_T[i])
        # 同频率正弦波的hankel矩阵进行SVD，并返回重构的hankel矩阵
        Rebuild_F = __SVD__(F_T_H, Fre_SVD_begin, Fre_SVD_end)
        Rebuild_S = __SVD__(S_T_H, Fre_SVD_begin, Fre_SVD_end)
        # 将降秩后的hankel矩阵反对角线平均化
        # 将重构hankel矩阵转化为单行数据
        T_Single_Row = __AntiDiagonAverage__(Rebuild_F)
        # 经过循环得到SVD并重构后的矩阵
        F_T[i] = T_Single_Row
        T_Single_Row = __AntiDiagonAverage__(Rebuild_S)
        S_T[i] = T_Single_Row
    #  endregion
    # 将一行表示(某一频率)，转换为一行表示(某一道)
    F = np.transpose(F_T)
    S = np.transpose(S_T)
    # 将该矩阵进行傅里叶反变换
    return __getIFFT__(F + 1j * S)

# endregion

# 对时域矩阵进行傅里叶变换
def __getFFT__(A):
    row = A.shape[0]
    col = A.shape[1]
    B = np.zeros((row, col))
    S = np.zeros((row, col))
    for i in range(row):
        fft_A = np.fft.fft(A[i])
        # fft_A_abs = np.abs(fft_A)
        # fft_A_angle = np.angle(fft_A)
        # B[i]= fft_A_abs
        # S[i]= fft_A_angle

        B[i] = np.real(fft_A)
        S[i] = np.imag(fft_A)
        # if i == 0:
        #     print('fft_A[0]')
        #     print(fft_A[0:10])
        #     print('B[0]:')
        #     print(B[0, 0:10])
        #     print('S[0]:')
        #     print(S[0, 0:10])

    return B, S


# 对频域矩阵进行傅里叶反变换
def __getIFFT__(F):
    row = F.shape[0]
    col = F.shape[1]
    B = np.zeros((row, col))
    for i in range(row):

        fft_F = np.fft.ifft(F[i])
        if i == 0:
            print('F[0]:')
            print(F[0, 0:10])
        fft_F = np.real(fft_F)
        B[i] = fft_F
    return B


# region 地震图绘制
def __Draw__(A):
    figure, ax = plt.subplots()
    # region 设置x，y值域
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    # endregion

    scalar = MaxAbsScaler()  # 加载函数
    scTempMat = scalar.fit_transform(A)  # 归一化
    print(scTempMat.shape)
    # endregion

    for j in range(0, A.shape[0]):
        Y_value = scTempMat[j] + 2 * j
        X = 2 * j
        i = 0
        X_value = [i for i in range(1, A.shape[1] + 1)]
        plt.plot((X, X), (0, 1500), linewidth='0.5', color='black')
        plt.plot(Y_value, X_value, linewidth='0.5', color='black')
        plt.fill_betweenx(X_value, Y_value, X, where=Y_value > X, facecolor='black', interpolate=True)
    plt.ylabel('t/ms')
    plt.xlabel('道号')
    plt.show()


# endregion

# region加载数据
def __LoadData__(name, row, col):
    print('Loading...Data...')
    A = np.zeros((row, col))
    A_row = 0
    src = 'Data/DataSource/'
    src = src + name
    fo = open(src, 'r')  # 打开数据文件文件
    lines = fo.readlines()

    for line in lines:  # 把lines中的数据逐行读取出来
        if A_row < row:
            List = line.strip().split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            A[A_row:] = List[0:col]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            A_row += 1  # 然后方阵A的下一行接着读
        else:
            break
    fo.close()
    print('End...__LoadData__')
    return A


# endregion

# region 单道SVD
def __SVD__(A, beginTime, endTime):
    pr = PrintTool()
    pr.isShow = False
    # pr.__print__('SVD...loading...')
    # pr.__print__('beginTime= '+str(beginTime)+'...endTime= '+str(endTime))
    # pr.__print__('The type of A is :')
    # pr.__print__(str(type(A)))
    # pr.__print__('The A is :')
    # pr.__print__(str(A))
    U, sigma, VT = np.linalg.svd(A)
    # print('sigma: '+str(len(sigma)))
    # 处理二维矩阵
    shape = np.shape(A)
    row = shape[0]
    col = shape[1]
    dig_len = len(sigma)
    pr.__print__('A:shape')
    pr.__print__(str(np.shape(A)))
    # dig = np.mat(np.eye(int(np.ceil(row * (endTime)) - np.ceil(row * beginTime) + 1)) * sigma[int(np.ceil(row * beginTime)) - 1:int(np.ceil(row * endTime))])
    dig = np.mat(np.eye(row, int(np.ceil(dig_len))) * sigma)

    pr.__print__('dig is :')
    pr.__print__(str(np.shape(dig)))
    # 获得对角矩阵
    redata = U[:, int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime))] * dig[int(
        np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), int(np.ceil(dig_len * beginTime)):int(
        np.ceil(dig_len * endTime))] * VT[int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), :]
    # redata = U * dig * VT
    Rebuilddata_cover = np.array(redata)
    # pr.__print__('Rebuilddata_cover is :')
    # pr.__print__(str(Rebuilddata_cover))
    # pr.__print__('SVD...ending...')
    return Rebuilddata_cover


# endregion


# region Hankel矩阵转换
def __SingalToHankel__(A):
    pr = PrintTool()
    pr.isShow = False
    n = len(A)
    string = 'The size of single A is :' + str(n)
    pr.__print__(string)
    pr.__print__('A type :' + str(type(A)))
    pr.__print__('A is :')
    pr.__print__(str(A))

    m = int(n / 2) + 1
    A = np.array(A)
    H = np.zeros((m, (n - m + 1)))

    for i in range(m):
        H[i] = A[i:n - m + i + 1]
    # H = scipy.linalg.hankel(A[:n-m+1],A[m:])
    pr.__print__('H type :' + str(type(H)))
    pr.__print__('H is :' + str(np.shape(H)))
    string = 'The hankel H is :\n' + str(H)
    pr.__print__(string)
    return H


# endregion

# hankel--->single
# region
def __HankeltoSingle__(A):
    m = A.shape[0]
    n = m + A.shape[1] - 1
    H = np.zeros((1, n))
    for i in range(A.shape[0]):
        H[0, i:n - m + i + 1] = A[i]
    return H[0]


# endregion
# region 频域图像
def __DrawFrequencyDomain__(A, x):
    col = len(A)
    # 设置需要采样的信号，频率分量有200，400和600
    # y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)

    plt.figure()
    plt.plot(x, A)
    plt.title('原始波形')
    # plt.show()

    # 快速傅里叶变换
    fft_A = fft(A)
    print(fft_A)
    '变换之后的结果数据长度和原始采样信号是一样的'
    '每一个变换之后的值是一个复数，为a+bj的形式'
    '复数a+bj在坐标系中表示为（a,b），故而复数具有模和角度'
    '快速傅里叶变换具有 “振幅谱”“相位谱”，它其实就是通过对快速傅里叶变换得到的复数结果进一步求出来的'
    '那这个直接变换后的结果是需要的，在FFT中，得到的结果是复数'
    'FFT得到的复数的模（即绝对值）就是对应的“振幅谱”，复数所对应的角度，就是所对应的“相位谱”'

    # FFT的原始频谱
    # 取复数的绝对值，即复数的模(双边频谱)
    abs_A = np.abs(fft_A)
    # 取复数的角度
    angle_A = np.angle(fft_A)

    plt.figure()
    plt.plot(x, abs_A)
    plt.title('双边振幅谱（未归一化）')

    plt.figure()
    plt.plot(x, angle_A)
    plt.title('双边相位谱（未归一化）')

    '我们在此处仅仅考虑“振幅谱”，不再考虑相位谱。'
    '我们发现，振幅谱的纵坐标很大，而且具有对称性，这是怎么一回事呢？'
    '关于振幅值很大的解释以及解决办法——归一化和取一半处理'
    '''
    比如有一个信号如下：
    Y=A1+A2*cos(2πω2+φ2）+A3*cos(2πω3+φ3）+A4*cos(2πω4+φ4）
    经过FFT之后，得到的“振幅图”中，
    第一个峰值（频率位置）的模是A1的N倍，N为采样点，本例中为N=1400，此例中没有，因为信号没有常数项A1
    第二个峰值（频率位置）的模是A2的N/2倍，N为采样点，
    第三个峰值（频率位置）的模是A3的N/2倍，N为采样点，
    第四个峰值（频率位置）的模是A4的N/2倍，N为采样点，
    依次下去......
    考虑到数量级较大，一般进行归一化处理，既然第一个峰值是A1的N倍，那么将每一个振幅值都除以N即可
    FFT具有对称性，一般只需要用N的一半，前半部分即可。
    '''

    # 归一化
    normalization_A = abs_A / col
    plt.figure()
    plt.plot(x, normalization_A, 'g')
    plt.title('双边频谱(归一化)', fontsize=9, color='green')

    # 取半处理
    half_x = x[range(int(col / 2))]  # 取一半区间
    normalization_half_A = normalization_A[range(int(col / 2))]  # 由于对称性，只取一半区间（单边频谱）
    plt.figure()
    plt.plot(half_x[0:50], normalization_half_A[0:50], 'b')
    plt.title('单边频谱(归一化)', fontsize=9, color='blue')


# endregion
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class PrintTool():
    isShow = True

    def __print__(self, str):
        strl = '\n' + str
        if self.isShow == True:
            print(strl)
