import os
import json
import warnings
import numpy as np
import torch


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def reid(bytes_rate):
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    gallery_fea_dir = 'gallery_feature'  #####################
    # gallery_fea_dir = 'extract_feature'
    reid_results_path = 'reid_results/{}.json'.format(bytes_rate)
    os.makedirs(os.path.dirname(reid_results_path), exist_ok=True)

    query_names = os.listdir(reconstructed_query_fea_dir)
    gallery_names = os.listdir(gallery_fea_dir)
    query_num = len(query_names)
    gallery_num = len(gallery_names)
    assert (query_num != 0 and gallery_num != 0)
    reconstructed_query_fea_list = []
    gallery_fea_list = []
    for query_name in query_names:
        reconstructed_query_fea_list.append(
            read_feature_file(os.path.join(reconstructed_query_fea_dir, query_name))
        )
    for gallery_name in gallery_names:
        gallery_fea_list.append(
            read_feature_file(os.path.join(gallery_fea_dir, gallery_name))
        )

    reconstructed_query_fea_all = np.stack(reconstructed_query_fea_list, axis=0).reshape((query_num, -1))
    gallery_fea_all = np.stack(gallery_fea_list, axis=0).reshape((gallery_num, -1))
    m, n = reconstructed_query_fea_all.shape[0], gallery_fea_all.shape[0]
    reconstructed_query_fea_all = torch.tensor(reconstructed_query_fea_all)
    gallery_fea_all = torch.tensor(gallery_fea_all)
    dist_m = torch.pow(reconstructed_query_fea_all, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(gallery_fea_all, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, reconstructed_query_fea_all, gallery_fea_all.t())  ## [m ,n]
    dist_m = dist_m.data.numpy()
    del reconstructed_query_fea_all  ## 防止cpu爆掉
    del gallery_fea_all
    gallery_array = np.asarray(gallery_names)
    indices = np.argsort(dist_m, axis=1)
    indices = indices[:, :100]
    data = dict()
    for i in range(query_num):
        data[query_names[i]] = gallery_array[indices[i]]
    with open(reid_results_path, 'w', encoding='UTF8') as f:
        save_submit_path = json.dumps(data, indent=2, cls=NpEncoder)
        f.write(save_submit_path)
    print('ReID Done ' + str(bytes_rate))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    br = [64, 128, 256]
    for byte in br:
        reid(byte)
