import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        for name, module in self.submodule._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                return x


'''
def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()  # 计算标准差
    if torch.cuda.is_available():
        eps = torch.FloatTensor(std.size()).normal_().cuda()  # 从标准的正态分布中随机采样一个eps
    else:
        eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)
'''


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def extract_feature(net, im_path: str) -> np.ndarray:
    # im = Image.open(im_path)
    img = cv2.imread(im_path)  # 加载图片
    img = cv2.resize(img, (224, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5])])
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize([0.485, 0.456, 0.406],
    #                           [0.229, 0.224, 0.225])])
    img = transform(img).cuda()
    img = img.unsqueeze(0)

    model = FeatureExtractor(net, ['layer4'])  # 指定提取 layer4 层特征
    with torch.no_grad():
        out = model(img)
        y2 = out.squeeze()
        out1 = nn.AdaptiveAvgPool2d((1, 1))
        y3 = out1(y2)
        fea = y3.squeeze()  # tensor
        # fea = Z_ScoreNormalization(fea, fea.mean(), fea.std())
        # mu, logvar = fea, fea
        # fea = reparametrize(mu, logvar)
        fea = fea.cpu().numpy()
        fea = np.asarray(fea).astype('<f4') / 255
        # fea = np.asarray(fea)[::4, ::4, 2].reshape(-1).astype('<f4') / 255
    return fea


def extract():
    img_dir = 'train_4/train_picture'
    fea_dir = 'extract_feature'
    # img_dir = 'image'  ####################
    # fea_dir = 'feature'  ####################
    os.makedirs(fea_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    assert (len(img_paths) != 0)

    net = models.resnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 15000)
    net.load_state_dict(torch.load('project/model-best.pth'))
    net.fc = net.fc.cuda()
    net = net.cuda()
    net.eval()

    for im_path in tqdm(img_paths):
        basename = get_file_basename(im_path)
        fea = extract_feature(net, im_path)
        write_feature_file(fea, os.path.join(fea_dir, basename + '.dat'))

    print('Extraction Done')


if __name__ == '__main__':
    extract()
