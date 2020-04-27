import torch
import numpy as np

from networks import LVNet
from df_dataset_visible import DFDatasets


# DEBUG_MODE = True
# use_gpu = True
# ckpt_path = 'checkpoints/v15/epoch_80.pth'
# test_txt = 'data/upper/train_list.txt'
# bbox_txt = 'data/Anno/list_bbox.txt'
#
# # load model
# model = LVNet()
# if use_gpu:
#     model.cuda()
# model.load_state_dict(torch.load(ckpt_path))
#
# # load data list
# test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE)
# test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)
#
# error_total = 0
# # predict
# for i, inputs in enumerate(test_loader):
#     x_gts, y_gts, conf_vis_gts = [n[0].cpu().numpy() for n in inputs['label_gt']]
#
#     im_tensor = inputs['im_tensor'].cuda()
#     output = model(im_tensor)
#     out_coord, out_conf_nocut, out_conf_viss = [n[0] for n in output]
#     out_xs, out_ys = [n.cpu().detach().numpy() for n in torch.split(out_coord, 6, dim=0)]
#
#     n_vibible = 0
#     d = 0
#     for nth_kpt in range(6):
#         conf_vis_gt = conf_vis_gts[nth_kpt]
#         x_gt = x_gts[nth_kpt]
#         y_gt = y_gts[nth_kpt]
#         out_x = out_xs[nth_kpt]
#         out_y = out_ys[nth_kpt]
#
#         if 1:
#             n_vibible += 1
#             dx = np.power(x_gt - out_x, 2)
#             dy = np.power(y_gt - out_y, 2)
#             d = d + dx + dy
#
#     error = d / n_vibible
#     error_total += error
#     print(' [*] Evaluate: {} / {}'.format(i, len(test_loader)))
#
# error_avg = error_total / len(test_loader)
# print(' [*] Average NMSE = {}'.format(error_avg))


def calculate_NMSE(gts, pds):
    gt_vis, gt_xs, gt_ys = gts
    out_vis, out_xs, out_ys = pds
    error = (np.sum(np.power(gt_xs - out_xs, 2)) + np.sum(np.power(gt_ys-out_ys, 2))) / 6
    return error


def evaluate_visible(gts, pds):
    pds = (pds > 0.5).astype('int')
    acc = (pds == gts)
    acc = acc.astype('int')
    acc = np.sum(acc)
    return acc





