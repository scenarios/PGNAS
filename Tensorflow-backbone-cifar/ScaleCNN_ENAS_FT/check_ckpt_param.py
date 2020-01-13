#!/usr/bin/env python

import sys
import tensorflow as tf
import numpy as np

def count(ckpt_fpath):
    # Open TensorFlow ckpt
    reader = tf.train.NewCheckpointReader(ckpt_fpath)

    print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
    param_map = reader.get_variable_to_shape_map()
    total_count = 0
    subnet_total_count = 0
    total_drop = 0.0
    ps_droprate = 0.7
    for k, v in param_map.items():
        if 'Momentum' not in k and 'global_step' not in k: #and 'batch_norm' in k: #'resnet_model/block_2/sub_block_2/bottleneck_2/kernel_size_3/weight' in k:
        #if 'Momentum' not in k and 'projection' in k:
            '''
            if 'weight' in k:
                k_pfx = k[::-1]
                k_pfx = k_pfx.split('/', 1)[-1][::-1]
                k_p = k_pfx+'/dropout_p_logit'
                p = reader.get_tensor(k_p)[0]
                p = 1. / (1. + np.exp(-p))

                temp = np.prod(v)*(1-p)
                total_count += temp
                print('%s: %s => %d' % (k, str(v), temp))
            elif 'bottleneck_3/conv2d/kernel' in k and 'root' not in k:
                k_pfx = k[::-1]
                k_pfx = k_pfx.split('/', 2)[-1][::-1]
                k_p_1 = k_pfx[:-1]+'2' + '/kernel_size_1/dropout_p_logit'
                k_p_3 = k_pfx[:-1]+'2' + '/kernel_size_3/dropout_p_logit'
                k_p_5 = k_pfx[:-1]+'2' + '/kernel_size_5/dropout_p_logit'
                k_p_9 = k_pfx[:-1]+'2' + '/kernel_size_9/dropout_p_logit'
                p_1 = reader.get_tensor(k_p_1)[0]
                p_1 = 1. / (1. + np.exp(-p_1))
                p_3 = reader.get_tensor(k_p_3)[0]
                p_3 = 1. / (1. + np.exp(-p_3))
                p_5 = reader.get_tensor(k_p_5)[0]
                p_5 = 1. / (1. + np.exp(-p_5))
                p_9 = reader.get_tensor(k_p_9)[0]
                p_9 = 1. / (1. + np.exp(-p_9))

                count = 0
                if p_1 > 0.95:
                    count += 1
                if p_3 > 0.95:
                    count += 1
                if p_5 > 0.95:
                    count += 1
                if p_9 > 0.95:
                    count += 1

                temp = np.prod(v) * (1 - count * 0.25)
                total_count += temp
                print('%s: %s => %d (%d)' % (k, str(v), temp, np.prod(v)))
            else:
                temp = np.prod(v)
                total_count += temp
                _ = reader.get_tensor(k)
                print('%s: %s => %d' % (k, str(v), temp))
             
            # For shake-shake
            if 'bn' not in k:
                temp = np.prod(v)
                total_count += temp
                if 'conv2/weights' in k:
                    sk = k.replace("weights", "dropout_p_logit")
                    _ = reader.get_tensor(sk)
                    subnet_total_count += temp*ps_droprate#(1 - (1. / (1. + np.exp(-_[0]))))
                elif 'conv1/weights' in k:
                    sk = k.replace("weights", "dropout_p_logit")
                    _1 = reader.get_tensor(sk)
                    
                    sk = k.replace("conv1/weights", "conv2/dropout_p_logit")
                    _2 = reader.get_tensor(sk)
                    subnet_total_count += temp*ps_droprate*ps_droprate#(1 - (1. / (1. + np.exp(-_1[0]))))*(1 - (1. / (1. + np.exp(-_2[0]))))
                if 'dropout' in k:
                    _ = reader.get_tensor(k)
                    print(1. / (1. + np.exp(-_[0])))
                    print('%s: %s => %d' % (k, str(v), temp))
            '''

            if  'dropout' not in k and 'bn' not in k:
                temp = np.prod(v)
                total_count += temp
                print('%s: %s => %d' % (k, str(v), temp))
                print(np.sum(reader.get_tensor(k)))
                if 'w' in k:
                    if 'depth' in k:
                        sk = k.replace("w_depth", "w_point")
                        _ = reader.get_tensor(sk+'_dropout_p_logit')
                        subnet_total_count += temp*ps_droprate#(1 - (1. / (1. + np.exp(-_[0]))))
                    else:
                        _ = reader.get_tensor(k+'_dropout_p_logit')
                        subnet_total_count += temp*ps_droprate#(1 - (1. / (1. + np.exp(-_[0]))))
                #print(1. / (1. + np.exp(-_[0])))
                #total_drop += 1. / (1. + np.exp(-_[0]))
            
    print('Total Param Count: %d' % total_count)
    print('Subnet Total Param Count: %f' % subnet_total_count)

if __name__=='__main__':
    count("/mnt/log/NAS/cifar10/enas/selection/E-I-random/model.ckpt-437850")
    #count("D:\workspace\yizhou/train\ScaleCNN\imagenet/res50\WarmupOriginalElgk3-2_kbase-2Ksizeallscaleequal\model.ckpt-1")
    #count("D:\workspace\yizhou/train\ScaleCNN\imagenet/res50/model_size_checker/model.ckpt-480385")
