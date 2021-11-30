"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
from torch.cuda import random
# from Model.config import args
import torch.nn.functional as F
import math
import random
import itertools

# from scipy import stats
# stats.entropy([0.95,0.05], base = 2)
#
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.metrics import mutual_info_score
# a = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
# b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
#
# print(stats.entropy([0.5,0.5])) # entropy of 0.69, expressed in nats
# print(mutual_info_classif(a.reshape(-1,1), b, discrete_features = True)) # mutual information of 0.69, expressed in nats
# print(mutual_info_score(a,b)) # information gain of 0.69, expressed in nats

simfunctions = {
    "euclidean" : lambda x, y: -torch.norm(x - y, p=2, dim=1).mean(), # 1/m ∑(xi-yi)^2
    "L1"        : lambda x, y: -torch.norm(x - y, p=1, dim=1).mean(),
    "MSE"       : lambda x, y: -(x - y).pow(2).mean(),                # 根号[ ∑(xi-yi)^2 ]
    "L3"        : lambda x, y: -torch.norm(x - y, p=3, dim=1).mean(),
    "Linf"      : lambda x, y: -torch.norm(x - y, p=float("inf"), dim=1).mean(),
    "soft_corr" : lambda x, y: F.softplus(x*y).sum(axis=1),
    "corr"      : lambda x, y: (x*y).sum(axis=1),
    "cosine"    : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).mean(),
    "angular"   : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).acos().mean() / math.pi,
}


def gradient_loss(s, dim = 3, penalty='l2'):
    if dim == 3:
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0
    elif dim == 2:
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims # [9, 9, 9]
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu)) # (1 ,1, 9, 9, 9)
    pad_no = math.floor(win[0] / 2) # 4
    stride = [1] * ndims # [1 ,1, 1]
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win) # 9*9*9
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    I_var = I2_sum / win_size - u_I * u_I
    J_var = J2_sum / win_size - u_J * u_J
    cross = IJ_sum / win_size - u_I * u_J 
    return I_var, J_var, cross


class NMI:

    def __init__(self, bin_centers, vol_size, sigma_ratio = 0.5, max_clip = 5000, local=False, crop_background = False, patch_size = 1):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo

        If you use this loss function, please cite the following:

        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis

        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.local_mi if local else self.global_mi
        self.vol_bin_centers = torch.Tensor(bin_centers)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = torch.Tensor([1 / (2 * np.square(self.sigma))]).cuda()
        print("preterm", self.preterm)
        self.eps = 1e-07

    def local_mi(self, y_true, y_pred):
        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, self.num_bins]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute padding sizes
        patch_size = self.patch_size
        x, y, z = self.vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = list(itertools.chain(*pad_dims)).reverse()

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = torch.exp(- self.preterm * torch.sqrt_(F.pad(y_true, padding, mode = "constant") - vbc)) # IA ← exp((A[:, new_axis]-bin_centers[new_axis,:]).square())
        I_a /= torch.sum(I_a, -1, keepdim = True) # normalize(IA, axis = 1)

        I_b = torch.exp(- self.preterm * torch.sqrt_(F.pad(y_pred, padding, mode = "constant") - vbc))
        I_b /= torch.sum(I_b, -1, keepdim = True)

        I_a_patch = torch.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_a_patch = torch.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = torch.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])

        I_b_patch = torch.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_b_patch = torch.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = torch.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])

        # compute probabilities
        I_a_permute = I_a_patch.permute(0,2,1)
        pab = torch.bmm(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = torch.mean(I_a_patch, 0, keepdim=True)
        pb = torch.mean(I_b_patch, 0, keepdim=True)

        papb = torch.bmm(pa.permute(0,2,1), pb) + self.eps
        return torch.mean(torch.sum(torch.sum(pab * torch.log(pab/papb + self.eps), 1), 1))

    def global_mi(self, y_true, y_pred):
        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = torch.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = F.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.unsqueeze(torch.unsqueeze(y_pred, 0), 2)
            y_true = torch.unsqueeze(torch.unsqueeze(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, height x width x depth x chan, 1)
            y_true = torch.reshape(y_true, (-1, np.prod(y_true.shape[1:]).item()))
            y_true = torch.unsqueeze(y_true, 2)
            # y_true(1, 4194304,1)
            y_pred = torch.reshape(y_pred, (-1, np.prod(y_pred.shape[1:]).item()))
            y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(self.vol_bin_centers.shape[:]).item()]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()
        # vbc(1,1,157)

        # compute image terms
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        # I_a(1, vol_num, bin_num)
        I_a = I_a / torch.sum(I_a, -1, keepdim = True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, -1, keepdim = True)

        # compute probabilities
        I_a_permute = I_a.permute(0,2,1)
        pab = torch.bmm(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab = pab / nb_voxels
        pa = torch.mean(I_a, 1, keepdim = True)
        pb = torch.mean(I_b, 1, keepdim = True)

        papb = torch.bmm(pa.permute(0,2,1), pb) + self.eps
        return torch.mean(torch.sum(torch.sum(pab * torch.log(pab/papb + self.eps), 1), 1))

    def loss(self, y_true, y_pred):
        y_pred = torch.clip(y_pred, 0, self.max_clip)
        y_true = torch.clip(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred)


def generate_patch_img(img, patch_size = 9, RANDOM = True, patch_num = 20):
        vol_size = img.shape[2:]
        patch_x_list, patch_y_list = choose_patch(img_size = vol_size, patch_size = patch_size, RANDOM = True, patch_num = patch_num)
        patch_img_list = []
        for i in range(len(patch_x_list)):
            patch_img = img[:, :, patch_x_list[i]: patch_x_list[i] + patch_size, patch_y_list[i]: patch_y_list[i] + patch_size]
            patch_img_list.append(patch_img)
        return torch.cat(patch_img_list, dim = 0)

def choose_patch(img_size, patch_size = 9, RANDOM = True, patch_num = 20):
    x_size = img_size[0]
    y_size = img_size[1]
    if RANDOM:
        # 产生 patch_num 个不重复的随机数
        patch_x_list = random.sample(range(0, x_size - patch_size), patch_num)
        patch_y_list = random.sample(range(0, y_size - patch_size), patch_num)      
    else:
        patch_num_x = math.floor(x_size / patch_size)  
        patch_num_y = math.floor(y_size / patch_size)
        patch_x_list = range(x_size / 2 - patch_num_x / 2 * patch_size, x_size / 2 + patch_num_x / 2 * patch_size, patch_size)
        patch_y_list = range(y_size / 2 - patch_num_y / 2 * patch_size, y_size / 2 + patch_num_y / 2 * patch_size, patch_size)
    return patch_x_list, patch_y_list