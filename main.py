# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_model():
    pthfile = r'model-64.pth'
    net = torch.load(pthfile)
    print(net)


def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def cal_d(x, x_hat):
    tmp = x - x_hat

    def square(x):
        return x ** 2

    tmp = map(square, tmp)
    tmp = [i for i in tmp]
    total = 0
    for ele in range(0, len(tmp)):
        total = total + tmp[ele]
    return np.sqrt(total)


def draw_plt(data):
    # plt.hist(data)
    n1, bins1, patches1 = plt.hist(data, bins=50, density=True, color='g', alpha=1)
    # 显示横轴标签
    plt.xlabel("range")
    # 显示纵轴标签
    plt.ylabel("count")
    # 显示图标题
    plt.xlim(-1, 1)
    plt.ylim(0, 25)
    # plt.title("频数/频率分布直方图")
    plt.show()


def xxx(br):
    # print('\n------%d------' % (br))
    fea_dir_1 = 'extract_feature'
    fea_dir_2 = 'reconstructed_query_feature/{}'.format(br)
    fea_paths_1 = glob.glob(os.path.join(fea_dir_1, '*.*'))
    fea_paths_2 = glob.glob(os.path.join(fea_dir_2, '*.*'))
    # print(len(fea_paths_1))
    # print(len(fea_paths_2))
    eval_avg = []
    for path_x, path_x_hat in tqdm(zip(fea_paths_1, fea_paths_2)):
        fea_x = read_feature_file(path_x)
        fea_x_hat = read_feature_file(path_x_hat)
        # draw_plt(fea_x)
        # draw_plt(fea_x_hat)
        eval_now = cal_d(fea_x, fea_x_hat)
        eval_avg.append(eval_now)
        # print('\nd = {}'.format())
        # print('x\t', np.max(fea_x), np.min(fea_x))
        # print('x_hat\t', np.max(fea_x_hat), np.min(fea_x_hat))
    total = 0
    for ele in range(0, len(eval_avg)):
        total = total + eval_avg[ele]
    Avg = total / len(eval_avg)
    print("[%d] average eval error:%f" % (br, Avg))


def compare_with_simple():
    fea_dir_1 = 'extract_feature'
    fea_dir_2 = 'query_feature_sample'
    fea_paths_1 = glob.glob(os.path.join(fea_dir_1, '*.*'))
    fea_paths_2 = glob.glob(os.path.join(fea_dir_2, '*.*'))
    for path_x in fea_paths_1:
        fea_x = read_feature_file(path_x)
        eval_avg = []
        for path_x_sim in fea_paths_2:
            fea_x_sim = read_feature_file(path_x_sim)
            eval_now = cal_d(fea_x, fea_x_sim)
            eval_avg.append(eval_now)
        total = 0
        for ele in range(0, len(eval_avg)):
            total = total + eval_avg[ele]
        Avg = total / len(eval_avg)
        print("average eval error:%f" % (Avg))


def read_fea(path):
    fea_paths = glob.glob(os.path.join(path, '*.*'))
    for fea_path in fea_paths:
        fea = read_feature_file(fea_path)
        # print(fea.shape)
        print(fea)
        print(np.max(fea), np.min(fea))
        draw_plt(fea)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # path = 'query_feature_sample'
    path = 'extract_feature'
    read_fea(path)
    exit()
    br = [64, 128, 256]
    for byte in br:
        xxx(byte)
    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
