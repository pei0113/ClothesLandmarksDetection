import random
import torch
import numpy as np
from termcolor import cprint
from time import time


# loss_arr = ['total', 'coord', 'nocut', 'cut', 'vis', 'occ']
# loss_arr = ['total', 'coord', 'nocut', 'vis']
# loss_arr = ['total', 'coord', 'vis']
# loss_arr = ['total', 'heat', 'vis']


def update_loss(batch_loss_dict, epoch_loss_dict, loss_arr):
    for i in loss_arr:
        epoch_loss_dict[i] += batch_loss_dict[i]
    return epoch_loss_dict


def show_epoch_loss(MODE, epoch_loss_dict, n, writer, epoch, tStart, loss_arr):
    log = '==>>> **{}** '.format(MODE)
    avg_loss_list = []
    for i, str in enumerate(loss_arr):
        avg_loss_list.append(epoch_loss_dict[str] / n)
        writer.add_scalar('{}/loss_{}'.format(MODE, str), avg_loss_list[i], epoch)
        log += 'loss_{} : {:.6f}  |  '.format(str, avg_loss_list[i])

    if MODE == 'train':
        cprint('='*70 + 'Epoch {}'.format(epoch+1)+ '='*70, color='yellow')
    print(log)

    if MODE == 'valid':
        return avg_loss_list[0]


def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    # k1 = multivariate_normal(mean=m1, cov=593.109206084)
    k1 = multivariate_normal(mean=m1, cov=s1)
    #     zz = k1.pdf(array_like_hm)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width))
    return img


def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)


def set_random_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic=True
    return print('[*] Set random seed: {}'.format(n))