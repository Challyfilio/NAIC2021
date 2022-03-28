import os
import glob
import numpy as np
import torch
from project.AutoEncoder import *
from tqdm import tqdm
import warnings


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


'''
def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    # assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    assert fea.ndim == 1 and fea.dtype == np.float32
    # fea.astype('<f4')[: target_bytes // 4].tofile(path)
    fea.astype('<f4')[: target_bytes].tofile(path)
    return True
'''


# old
def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(int(fea.shape[0]).to_bytes(4, byteorder='little', signed=False))
        f.write(fea.astype('<f4')[:(target_bytes - 4) // 4].tostring())
    return True


def compress(bytes_rate):
    # if not isinstance(bytes_rate, int):
    bytes_rate = int(bytes_rate)
    query_fea_dir = 'extract_feature'
    # query_fea_dir = 'query_feature'  ####################
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)
    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    # -----随机打乱，取1.5w-----
    # np.random.shuffle(query_fea_paths)
    # query_fea_paths = query_fea_paths[0:15000] #切片
    # print(len(query_fea_paths))
    # ----------
    assert (len(query_fea_paths) != 0)

    Coder = AutoEncoder(bytes_rate)
    if bytes_rate == 64:
        Coder.load_state_dict(torch.load('project/model-test-ar-A64.pth'))
    elif bytes_rate == 128:
        Coder.load_state_dict(torch.load('project/model-test-ar-A128.pth'))
    elif bytes_rate == 256:
        Coder.load_state_dict(torch.load('project/model-test-ar-A256.pth'))
    else:
        pass
    Coder = Coder.cuda()
    Coder.eval()

    for query_fea_path in tqdm(query_fea_paths):
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        # print('输入的维度', fea.shape)
        # print(fea)
        # -----
        fea = torch.from_numpy(fea).cuda()
        fea, _ = Coder(fea, tag=False)
        # print('压缩的维度', fea.size())
        # print(fea)
        fea = fea.cpu()
        fea = fea.detach().numpy()
        # print('输出的结果的维度', fea.shape)
        # print(fea.shape)

        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        compress_feature(fea, bytes_rate, compressed_fea_path)

    print('Compression Done ' + str(bytes_rate))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    br = [64, 128, 256]
    for byte in br:
        compress(byte)
