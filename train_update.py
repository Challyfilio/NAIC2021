import os
import glob
import time
import torch
import shutil
import numpy as np
from project.AutoEncoder import *
from tqdm import tqdm
import warnings
from torch.autograd import Variable


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

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(int(fea.shape[0]).to_bytes(4, byteorder='little', signed=False))
        f.write(fea.astype('<f4')[:(target_bytes - 4) // 4].tostring())
    return True


def decompress_feature(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        feature_len = int.from_bytes(f.read(4), byteorder='little', signed=False)
        fea = np.frombuffer(f.read(), dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(feature_len - fea.shape[0], dtype='<f4')], axis=0
    )
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
    loss = np.sqrt(total)
    loss = torch.tensor(loss)
    loss = Variable(loss, requires_grad=True)
    return loss


def train(EPOCH, LR, byte):
    torch.manual_seed(1)
    starttime = time.time()
    # ----- start -----
    test_gallery_path = np.array(glob.glob(r'../NAICreid/datasets/NAIC2021Reid/train_feature/*.dat'))  # 259450
    # test_gallery_path = np.array(glob.glob(r'../NAICreid/datasets/NAIC2021Reid/query_feature_A/*.dat'))
    # test_gallery_path = np.array(glob.glob('test/*.dat'))
    np.random.shuffle(test_gallery_path)
    # print(test_gallery_path)
    # print(test_gallery_path.shape)
    test_gallery_path = test_gallery_path[0:50000, ]  # 切片
    # print(test_gallery_path)
    # print(test_gallery_path.shape)
    # exit()

    Coder = AutoEncoder(byte)
    Coder = Coder.cuda()
    Coder.train()
    print(byte)
    # print(Coder)
    optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    train_loss_avg = []
    eval_avg = []
    print('Training...')
    for epoch in range(EPOCH):
        train_loss_avg.append(0)
        eval_avg.append(0)
        num_batches = 0
        loss_best = 10
        compressed_query_fea_dir = 'temp'
        os.makedirs(compressed_query_fea_dir, exist_ok=True)
        for path in tqdm(test_gallery_path):
            query_basename = get_file_basename(path)
            features = read_feature_file(path)
            # print(features.shape)
            # print(features)
            input = torch.from_numpy(features).cuda()
            # print(input.shape)
            # print(input)
            encode, _ = Coder(input, tag=False)
            # encode = encode.cpu().numpy()
            encode = encode.cpu()
            encode = encode.detach().numpy()
            # print(encode.shape)
            # print(encode)
            # save file
            compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
            compress_feature(encode, byte, compressed_fea_path)
            # -----compress finish-----

            compressed_query_fea_file = 'temp/{}.dat'.format(query_basename)
            encode_temp = decompress_feature(compressed_query_fea_file)
            # print(encode_temp.shape)
            # print(encode_temp)
            encode_temp = torch.from_numpy(encode_temp).cuda()
            _, decode = Coder(encode_temp, tag=True)
            # print(decode.shape)
            # print(decode)
            # -----reconstruct finish-----

            loss = loss_func(decode, input).cuda()
            input = input.cpu()
            decode = decode.cpu()
            eval_now = cal_d(input.detach().numpy(), decode.detach().numpy()).cuda()

            # print(loss)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            # one step of the optimizer(using the gradients form backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            eval_avg[-1] += eval_now.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        eval_avg[-1] /= num_batches
        print("Epoch [%d  / %d] average reconstruction error:%f eval error:%f" % (
            epoch + 1, EPOCH, train_loss_avg[-1], eval_avg[-1]))
        shutil.rmtree(compressed_query_fea_dir)
        if train_loss_avg[-1] < loss_best:
            torch.save(Coder.state_dict(), 'model-test-relu-A{}.pth'.format(byte))  # 只保存模型的参数

    # torch.save(Coder.state_dict(), 'model-test-B{}.pth'.format(byte))  # 只保存模型的参数
    print('________________________________________')
    print('Finish Training')

    # ----- end -----
    endtime = time.time()
    print('训练耗时：{}\n'.format(endtime - starttime))


def htest(byte):
    # ----- start -----
    # test_gallery_path = np.array(glob.glob(r'../NAICreid/datasets/NAIC2021Reid/train_feature/*.dat'))  # 259450
    # test_gallery_path = np.array(glob.glob(r'../NAICreid/datasets/NAIC2021Reid/query_feature_A/*.dat'))
    # test_gallery_path = np.array(glob.glob(r'extract_feature/*.dat'))
    test_gallery_path = np.array(glob.glob(r'extract_feature/*.dat'))
    np.random.shuffle(test_gallery_path)
    # test_gallery_path = test_gallery_path[0:50000, ] #切片
    compressed_query_fea_dir = 'temp'
    os.makedirs(compressed_query_fea_dir, exist_ok=True)
    eval_avg = []
    # print(Coder)
    Coder = AutoEncoder(byte)
    pretrain_model = 'model-test-ar-A{}.pth'.format(byte)
    Coder.load_state_dict(torch.load(pretrain_model))
    Coder = Coder.cuda()
    Coder.eval()
    # if byte == 64:
    #     Coder.load_state_dict(torch.load('model-test-drop-A64.pth'))
    # elif byte == 128:
    #     Coder.load_state_dict(torch.load('model-test-drop-A128.pth'))
    # elif byte == 256:
    #     Coder.load_state_dict(torch.load('model-test-drop-A256.pth'))
    # else:
    #     pass
    for path in tqdm(test_gallery_path):
        query_basename = get_file_basename(path)
        features = read_feature_file(path)
        input = torch.from_numpy(features).cuda()
        encode, _ = Coder(input, tag=False)
        encode = encode.cpu()
        encode = encode.detach().numpy()
        # save file
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        compress_feature(encode, byte, compressed_fea_path)
        # -----compress finish-----
        compressed_query_fea_file = 'temp/{}.dat'.format(query_basename)
        encode_temp = decompress_feature(compressed_query_fea_file)
        encode_temp = torch.from_numpy(encode_temp).cuda()
        _, decode = Coder(encode_temp, tag=True)
        # -----reconstruct finish-----
        input = input.cpu()
        decode = decode.cpu()
        eval_now = cal_d(input.detach().numpy(), decode.detach().numpy()).cuda()
        eval_avg.append(eval_now)

    total = 0
    for ele in range(0, len(eval_avg)):
        total = total + eval_avg[ele]
    Avg = total / len(eval_avg)
    print("[%d] average eval error:%f" % (byte, Avg))
    shutil.rmtree(compressed_query_fea_dir)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    EPOCH = 10
    LR = 0.0001
    # train(EPOCH, LR, 128)
    br = [64, 128, 256]
    for byte in br:
        # train(EPOCH, LR, byte)
        htest(byte)
