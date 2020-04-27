import torch
import numpy as np
from torch import nn

bias = 1e-15


def FOCAL(out, gt, r=2):
    out = out.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    # one_map = np.ones((gt.shape[0], gt.shape[1]))
    loss = np.mean(-((1-out)**r * gt * np.log(out+bias) + out**r * (1-gt) * np.log(1-out+bias)))
    return loss


def tensor_se(a, b):
    x = a - b
    x = torch.mul(x, x)
    return x


def MAE(out, gt, prob):
    n = torch.sum(prob)
    if n == 0:
        loss = 0
    else:
        loss = torch.sum(prob * torch.abs(out - gt)) / n
    return loss


def BCE(out, gt):
    out = out.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    one_map = np.ones((gt.shape[0], gt.shape[1]))
    loss = np.mean(-(gt*np.log(out) + (one_map-gt)*np.log(one_map-out)))
    return loss


def criterionBCE(output, x_gt, y_gt, conf_nocut, conf_vis):
    criterion = nn.BCELoss()
    out_coord, out_conf_nocut, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)
    # if no cut-off
    loss_x = torch.abs(out_x - x_gt)
    loss_y = torch.abs(out_y - y_gt)
    loss_coord = torch.sum(conf_nocut * (loss_x + loss_y)) / torch.sum(conf_nocut)
    # confidence no cut or cut
    loss_nocut = criterion(out_conf_nocut, conf_nocut)
    # confidence visible or occlded
    loss_visible = conf_nocut * criterion(out_conf_vis, conf_vis)

    loss_total = loss_coord + loss_nocut + loss_visible
    loss_dict = {'total': loss_total,
                 'coord': loss_coord,
                 'nocut': loss_nocut,
                 'vis': loss_visible}
    return loss_dict


def criterionFOCAL(output, x_gt, y_gt, conf_nocut, conf_vis, r):
    out_coord, out_conf_nocut, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)
    # if no cut-off
    loss_x = torch.abs(out_x - x_gt)
    loss_y = torch.abs(out_y - y_gt)
    loss_coord = torch.sum(conf_nocut * (loss_x + loss_y)) / torch.sum(conf_nocut)
    # confidence no cut or cut
    loss_nocut = FOCAL(out_conf_nocut, conf_nocut, r)
    # confidence visible or occlded
    loss_visible = FOCAL(out_conf_vis, conf_vis, r)

    loss_total = loss_coord + loss_nocut + loss_visible
    loss_dict = {'total': loss_total,
                 'coord': loss_coord,
                 'nocut': loss_nocut,
                 'vis': loss_visible}
    return loss_dict


def criterionFOCAL_vis(output, x_gt, y_gt, conf_vis, r):
    out_coord, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)
    # if no cut-off
    loss_x = torch.abs(out_x - x_gt)
    loss_y = torch.abs(out_y - y_gt)
    n = conf_vis.shape[0] * conf_vis.shape[1]
    loss_coord = torch.sum((loss_x + loss_y)) / n
    # confidence visible or occluded
    loss_visible = FOCAL(out_conf_vis, conf_vis, r)

    loss_total = loss_coord + loss_visible
    loss_dict = {'total': loss_total,
                 'coord': loss_coord,
                 'vis': loss_visible}
    return loss_dict


def criterionBCE_vis(output, x_gt, y_gt, conf_vis):
    criterion = nn.BCELoss()
    out_coord, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)
    # if no cut-off
    loss_x = torch.abs(out_x - x_gt)
    loss_y = torch.abs(out_y - y_gt)
    n = conf_vis.shape[0] * conf_vis.shape[1]
    loss_coord = torch.sum((loss_x + loss_y)) / n
    # confidence visible or occlded
    loss_visible = criterion(out_conf_vis, conf_vis)

    loss_total = loss_coord + loss_visible
    loss_dict = {'total': loss_total,
                 'coord': loss_coord,
                 'vis': loss_visible}
    return loss_dict


def criterionHEAT(output_heat, heat, out_vis, vis):
    criterion_heat = nn.MSELoss()
    criterion_vis = nn.BCELoss()
    loss_heat = criterion_heat(output_heat, heat)
    loss_vis = criterion_vis(out_vis, vis)
    loss_total = loss_heat + loss_vis
    loss_dict = {'total': loss_total,
                 'heat': loss_heat,
                 'vis': loss_vis}
    return loss_dict
