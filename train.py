# python imports
import os
import glob
import warnings
import time
# external imports
import torch
import numpy as np
import SimpleITK as sitk
import math
from torch.optim import Adam
import torch.utils.data as Data
import torch.nn.functional as F
import itertools
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators_affine import Dataset
from Model.model import U_Network, SpatialTransformer, AlexNet, MLP
from settings import setting

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dim = 2

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(setting['model_dir']):
        os.makedirs(setting['model_dir'])
    if not os.path.exists(setting['log_dir']):
        os.makedirs(setting['log_dir'])


def save_image(img, ref_img, name, if_flow = False):
    if if_flow == False:
        img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
        ref_img = sitk.GetImageFromArray(ref_img[0, 0, ...].cpu().detach().numpy())
        img.SetOrigin(ref_img.GetOrigin())
        img.SetDirection(ref_img.GetDirection())
        img.SetSpacing(ref_img.GetSpacing())
        sitk.WriteImage(img, os.path.join(setting['result_dir'], name))
    else:
        img = sitk.GetImageFromArray(img[0, ...].cpu().detach().numpy())
        sitk.WriteImage(img, os.path.join(setting['result_dir'], name))


def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(setting['gpu']) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_name = str(setting['feat_dim'])+"_"+str(setting['patch_num'])+"_"+str(setting['lr_unet'])+"_"+str(setting['lr_ext'])+'_'+str(setting['alpha'])+'_'+str(time_str)
    print("log_name: ", log_name)
    f = open(os.path.join(setting['log_dir'], log_name + ".txt"), "w")

    vol_size = (240, 240)
    sim_loss_fn = losses.simfunctions[setting["sim_fn"]]
    # bin_centers = range(-200, 5000, 100)
    # sim_loss_fn = losses.NMI(bin_centers = bin_centers, vol_size = vol_size).loss
    # 创建配准网络（UNet_m2f）和STN
    nf_enc = [16, 32, 32, 32]
    if setting['model']== "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet_m2f = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet_f2m = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN1 = SpatialTransformer(vol_size).to(device)
    STN2 = SpatialTransformer(vol_size).to(device)
    Extractor_m1 = MLP(input_size = setting["patch_size"]*setting["patch_size"], feat_dim = setting["feat_dim"]).to(device)
    Extractor_m2 = MLP(input_size = setting["patch_size"]*setting["patch_size"], feat_dim = setting["feat_dim"]).to(device)

    UNet_m2f.train()
    UNet_f2m.train()
    STN1.train()
    STN2.train()
    Extractor_m1.train()
    Extractor_m2.train()
    # 模型参数个数
    print("UNet_m2f: ", count_parameters(UNet_m2f))
    print("Extractor: ", count_parameters(Extractor_m1))
    print("STN: ", count_parameters(STN1))
    print("==========>")

    # Set opt_unetimizer and losses
    opt_unet = Adam(itertools.chain(UNet_m2f.parameters(), UNet_f2m.parameters()),lr = setting['lr_unet'])
    opt_ext = Adam(itertools.chain(Extractor_m1.parameters(), Extractor_m2.parameters()), lr = setting['lr_ext'])

    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data

    DS_train = Dataset(["../../Dataset/MICCAI_BraTS/Deformed2D/HGG/train/*/t1.nii.gz",
                  "../../Dataset/MICCAI_BraTS/Deformed2D/HGG/train/*/t2.nii.gz"])
    print("Number of training images: ", len(DS_train))
    DL_train = Data.DataLoader(DS_train, batch_size = setting['batch_size'], shuffle = True, num_workers = 4, drop_last = True)

    mod1 = slice(0, 240)
    mod2 = slice(240, 480)
    # Training loop.
    for i in range(1, setting['n_iter'] + 1):
        # Generate the moving images and convert them to tensors.
        imgs = iter(DL_train).next()
        # [B, C, D, W, H]
        input_moving = imgs[:, :, :, mod1].to(device).float()
        input_fixed = imgs[:, :, :,mod2].to(device).float()

        # Run the data through the model to produce warp and flow field
        torch.autograd.set_detect_anomaly(True)
        flow_m2f = UNet_m2f(input_moving, input_fixed)
        flow_f2m = UNet_f2m(input_fixed, input_moving)

        m2f = STN1(input_moving, flow_m2f)  # wraped_moving. M1. vs fixed
        f2m = STN2(input_fixed, flow_f2m)   # wraped_fixed.  M2. vs moving

        """ 选取 patch """
        input_moving_patch = losses.generate_patch_img(input_moving, patch_size=setting['patch_size'], RANDOM=True, patch_num=setting['patch_num'])
        input_fixed_patch = losses.generate_patch_img(input_fixed, patch_size=setting['patch_size'], RANDOM=True, patch_num=setting['patch_num'])
        f2m_patch = losses.generate_patch_img(f2m, patch_size=setting['patch_size'], RANDOM=True, patch_num=setting['patch_num'])
        m2f_patch = losses.generate_patch_img(m2f, patch_size=setting['patch_size'], RANDOM=True, patch_num=setting['patch_num'])

        """ 提取公共表示"""
        feature_m = Extractor_m1(input_moving_patch)
        feature_f = Extractor_m2(input_fixed_patch)
        feature_f2m = Extractor_m1(f2m_patch)
        feature_m2f = Extractor_m2(m2f_patch)

        """ 计算 Loss """
        sim_loss = - sim_loss_fn(feature_m, feature_f2m) - sim_loss_fn(feature_f, feature_m2f)
        grad_loss = grad_loss_fn(flow_m2f, dim) + grad_loss_fn(flow_f2m, dim)
        loss = sim_loss + setting['alpha']* grad_loss
        print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # sim_loss = sim_loss_fn(input_moving, f2m) + sim_loss_fn(input_fixed, m2f)
        # grad_loss = grad_loss_fn(flow_m2f, dim) + grad_loss_fn(flow_f2m, dim)
        # loss = sim_loss + setting['alpha']* grad_loss
        # print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        # print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        """ Backward """
        opt_unet.zero_grad()
        opt_ext.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        opt_unet.step()
        opt_ext.step()

        if i % setting['n_save_iter'] == 0:
            # Save model checkpoint
            save_file_name_unet1 = os.path.join(setting['model_dir'], '%d_unet1.pth' % i)
            save_file_name_unet2 = os.path.join(setting['model_dir'], '%d_unet2.pth' % i)
            save_file_name_ext1 = os.path.join(setting['model_dir'], '%d_ext1.pth' % i)
            save_file_name_ext2 = os.path.join(setting['model_dir'], '%d_ext2.pth' % i)
            print(Extractor_m1.state_dict())
            # print(Extractor_m2.state_dict())
            torch.save(UNet_m2f.state_dict(), save_file_name_unet1)
            torch.save(UNet_f2m.state_dict(), save_file_name_unet2)
            torch.save(Extractor_m1.state_dict(), save_file_name_ext1)
            torch.save(Extractor_m2.state_dict(), save_file_name_ext2)
            # Save images
            m_name = str(i) + "_m.nii.gz"
            f_name = str(i) + '_f.nii.gz'
            m2f_name = str(i) + "_m2f.nii.gz"
            f2m_name = str(i) + "_f2m.nii.gz"
            flow_m2f_name = str(i) + '_flow_m2f.nii.gz'
            save_image(input_moving, input_fixed, m_name)
            save_image(input_fixed, input_fixed, f_name)
            save_image(m2f, input_fixed, m2f_name)
            save_image(f2m, input_fixed, f2m_name)
            save_image(flow_m2f, input_fixed, flow_m2f_name, if_flow=True)
            print("warped images have saved.")
    f.close()


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = DeprecationWarning)
    train()
