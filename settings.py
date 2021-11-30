setting={}
setting['gpu'] = '0'

setting['model'] = 'vm2'
setting['feat_dim'] = 64
setting['patch_size'] = 30
setting['patch_num'] = 15

setting['batch_size'] = 8
setting['lr_unet'] = 4e-4
setting['lr_ext'] = 4e-4
setting['n_iter'] = 10000
setting['n_save_iter'] = 20
setting['alpha'] = 0.1

setting['model_dir'] = './Checkpoint'
setting['log_dir'] = './log'
setting['result_dir'] = './Result'
setting['sim_fn'] = 'cosine'

if __name__ == '__main__':
    print(setting)