import os
import glob
import numpy as np
import torch
from project.AutoEncoder import *
from tqdm import tqdm
import warnings


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def reconstruct_feature(path: str) -> np.ndarray:
    fea = np.fromfile(path, dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(2048 - fea.shape[0], dtype='<f4')], axis=0
    )
    return fea


def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def decompress_feature(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        feature_len = int.from_bytes(f.read(4), byteorder='little', signed=False)
        fea = np.frombuffer(f.read(), dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(feature_len - fea.shape[0], dtype='<f4')], axis=0
    )
    return fea


def reconstruct(bytes_rate):
    bytes_rate = int(bytes_rate)
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)
    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    assert (len(compressed_query_fea_paths) != 0)

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

    for compressed_query_fea_path in tqdm(compressed_query_fea_paths):
        query_basename = get_file_basename(compressed_query_fea_path)
        # reconstructed_fea = reconstruct_feature(compressed_query_fea_path)
        reconstructed_fea = decompress_feature(compressed_query_fea_path)
        # reconstructed_fea = read_feature_file(compressed_query_fea_path)
        # print('输入的维度', reconstructed_fea.shape)
        # print(reconstructed_fea)
        # -----
        reconstructed_fea = torch.from_numpy(reconstructed_fea).cuda()
        _, reconstructed_fea = Coder(reconstructed_fea, tag=True)
        # print('重建的维度', reconstructed_fea.size())
        # print(reconstructed_fea)
        reconstructed_fea = reconstructed_fea.cpu()
        reconstructed_fea = reconstructed_fea.detach().numpy()
        # print('输出的结果的维度', reconstructed_fea.shape)
        # print(reconstructed_fea)

        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Reconstruction Done ' + str(bytes_rate))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    br = [64, 128, 256]
    for byte in br:
        reconstruct(byte)
