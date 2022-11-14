import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [0.0] * dataset_size_test

            time_val = 0
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                if opt.calc_metrics:
                    psnr[i] = calc_psnr(res['data_gt'], res['data_out'])
                
                if opt.save_imgs:
                    file_name = data['fname'][0].split('-')
                    folder_dir = './ckpt/%s/output/%s' % (opt.name, file_name[0])  
                    os.makedirs(folder_dir, exist_ok=True)
                    save_dir = '%s/%s' % (folder_dir, file_name[1])
                    dataset_test.imio.write(np.array(res['data_out'][0].cpu()).astype(np.uint8), save_dir)

            avg_psnr = '%.2f'%np.mean(psnr)
            print('Time: %.3f s AVG Time: %.3f ms PSNR: %s\n' % (time_val, time_val/dataset_size_test*1000, avg_psnr))
    for dataset in datasets:
        datasets[dataset].close()
