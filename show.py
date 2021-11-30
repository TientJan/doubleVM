from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("agg")


""" show loss """
import math
def show_loss(log_name, x_lim=[0,100], y_lim=[-1,1], step=10):
    # 读取 log 信息
    loss_list = []
    loss_sim_list = []
    loss_grad_list = []
    with open(log_name, "r") as f:  # 打开文件
        for line in f.readlines():
            line = line.strip('\n').split(',')
            loss_list.append(float(line[1]))
            loss_sim_list.append(float(line[2]))
            loss_grad_list.append(float(line[3]))
    f.close()

    n_iter = len(loss_list)
    loss_average_list = [sum(loss_list[step*i : step*i + step]) / step for i in range(math.ceil(n_iter/step))]

    iter_list  = range(1, n_iter + 1)
    iter_average_list = range(1, n_iter + 1, step)
    plt.plot(iter_list, loss_list)
    # plt.plot(iter_list, loss_sim_list, 'g--')
    plt.plot(iter_list, loss_grad_list, 'y--')
    plt.plot(iter_average_list, loss_average_list, color='red')
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.savefig('figures/loss.jpg')

''' show warped result'''
import nibabel as nib
def show_result(iter):
    m = nib.load('Result/{}_m.nii.gz'.format(iter)).get_fdata()
    f = nib.load('Result/{}_f.nii.gz'.format(iter)).get_fdata()
    m2f = nib.load('Result/{}_m2f.nii.gz'.format(iter)).get_fdata()
    f2m = nib.load('Result/{}_f2m.nii.gz'.format(iter)).get_fdata()

    flow = nib.load('Result/{}_flow_m2f.nii.gz'.format(iter)).get_fdata()
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(m, cmap='gray')
    plt.title('t1')
    plt.subplot(2,2,2)
    plt.imshow(f, cmap='gray')
    plt.title('t2')
    plt.subplot(2,2,3)
    plt.imshow(m2f, cmap='gray')
    plt.title('warped t1')
    plt.subplot(2,2,4)
    plt.imshow(f2m, cmap='gray')
    plt.title('warped t2')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('figures/result.jpg')

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(flow[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(flow[:,:,1], cmap='gray')
    plt.savefig('figures/flow.jpg')

from Model.model import U_Network, MLP
import torch
from settings import setting
def show_net_parameter(iter):
    unet1 = U_Network()
    unet2 = U_Network()
    ext1 = MLP(feat_dim=setting['feat_dim'])
    ext2 = MLP(feat_dim=setting['feat_dim'])

    unet1.load_state_dict(torch.load('Checkpoint/{}_unet1.pth'.format(iter)))
    unet2.load_state_dict(torch.load('Checkpoint/{}_unet2.pth'.format(iter)))
    ext1.load_state_dict(torch.load('Checkpoint/{}_ext1.pth'.format(iter)))
    ext2.load_state_dict(torch.load('Checkpoint/{}_ext2.pth'.format(iter)))

if __name__ == '__main__':
    show_loss(log_name ='log/64_15_0.0004_0.0004_15_2021-11-18-23_43_32.txt', x_lim=[0,200], y_lim=[0,0.03], step=20)
    show_result(iter = 40)