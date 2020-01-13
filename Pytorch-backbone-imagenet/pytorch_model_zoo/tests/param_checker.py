import os

import numpy as np

import torch

CHECK_POINT_PATH = "D:/v-yizzh/train/NAS/imagenet/dense121/lr0.1_epc180_bs300_p0.1_tmp1to5_ls100_wd1/checkpoint.pth.tar"

def _compute_size(weight, drop_list, layer, name):
    in_c = weight.shape[1]
    ratio = (layer-1)*32.0/in_c
    if len(drop_list) > 1:
        print(name)
        print("{} -> {}, cut {}".format(weight.nelement(),
                                        weight.nelement() * (1 - (ratio * drop_list[1:].mean() + (1 - ratio) * drop_list[0])),
                                        1 - (ratio * drop_list[1:].mean() + (1 - ratio) * drop_list[0]))
              )
        return weight.nelement() * (1 - (ratio * drop_list[1:].mean() + (1 - ratio) * drop_list[0]))
    else:
        print(name)
        print("{} -> {}, cut {}".format(weight.nelement(),
                                        weight.nelement() * (1 - (ratio * 0 + (1 - ratio) * drop_list[0])),
                                        1 - (ratio * 0 + (1 - ratio) * drop_list[0]))
              )
        return weight.nelement() * (1 - (ratio *0 + (1 - ratio) * drop_list[0]))

def count_params(ckpt):
    o_total_size = 0
    s_total_size = 0
    param_dict = ckpt['state_dict']
    for name in param_dict.keys():
        if 'conv' in name and 'weight' in name and 'conv0' not in name:
            o_size = param_dict[name].nelement()
            s_size = _compute_size(
                param_dict[name],
                1./(1+np.exp(-param_dict[name.replace('weight', 'p_logit')])),
                layer = 1 if 'transition' in name else int(name.split('denselayer')[-1].split('.')[0]),
                name=name)

            o_total_size += o_size
            s_total_size += s_size

    return [s_total_size, o_total_size]


def main():
    checkpoint = torch.load(CHECK_POINT_PATH, map_location='cpu')

    print(count_params(checkpoint))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = " "
    main()