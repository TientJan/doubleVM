setting={}
setting['gpu'] = '0'

setting['model'] = 'vm2'

setting['batch_size'] = 8
setting['lr_unet'] = 3e-3
setting['lr_ext'] = 0.01
setting['n_iter'] = 10000
setting['n_save_iter'] = 100
setting['alpha'] = 20.0

setting['model_dir'] = './Checkpoint'
setting['log_dir'] = './log'
setting['result_dir'] = './Result'

if __name__ == '__main__':
    print(setting)