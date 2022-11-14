#!/bin/bash
echo "Start to test the model...."

dataroot="/Data/dataset/GOPRO_Large/"  # Modify the path of GOPRO dataset.

type="gauss5_50"  # or "poisson5_50"
# When experimenting on Gaussian noise, choose "gauss5_50". 
# When experimenting on Poisson noise, choose "poisson5_50".

name="selfir_gauss_noise"  # Folder name of the testing model

iter='1' # Epoch number of the testing model

device='0'  #  GPU id


python test.py \
    --dataset_name gopro    --model selfir     --noisetype $type       --name $name    --dataroot $dataroot  \
    --load_iter $iter       --save_imgs True   --calc_metrics True     --gpu_id $device   

python metrics.py  --name $name --dataroot $dataroot --device $device  