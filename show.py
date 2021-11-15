from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("agg")


""" show loss """
import math

# 读取 log 信息
loss_list = []
loss_sim_list = []
loss_grad_list = []
with open("log/10000_0.003_0.01_20.0_2021-11-11-23_33_54.txt", "r") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n').split(',')
        loss_list.append(float(line[1]))
        loss_sim_list.append(float(line[2]))
        loss_grad_list.append(float(line[3]))
f.close()

n_iter = len(loss_list)
step = 10
loss_average_list = [sum(loss_list[step*i : step*i + step]) / step for i in range(math.ceil(n_iter/step))]

iter_list  = range(1, n_iter + 1)
iter_average_list = range(1, n_iter + 1, step)
plt.plot(iter_list, loss_list)
# plt.plot(iter_list, loss_sim_list, 'g--')
# plt.plot(iter_list, loss_grad_list, 'y--')
plt.plot(iter_average_list, loss_average_list, color='red')
plt.ylim([-2.3, 1])
plt.xlim([0, 800])
plt.savefig('figures/loss.jpg')

''' show warped result'''
import nibabel as nib

m = nib.load('Result/200_m.nii.gz').get_fdata()
f = nib.load('Result/200_f.nii.gz').get_fdata()
m2f = nib.load('Result/200_m2f.nii.gz').get_fdata()
f2m = nib.load('Result/200_f2m.nii.gz').get_fdata()

flow = nib.load('Result/200_flow_m2f.nii.gz').get_fdata()
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

print(flow.shape)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(flow[:,:], cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(flow[:,:,1], cmap='gray')
plt.savefig('figures/flow.jpg')