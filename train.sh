#!/bin/bash
echo "Start to train the model...."

dataroot="/Data/dataset/GOPRO_Large/"  # Modify the path of GOPRO dataset.

type="gauss5_50"  # or "poisson5_50"
# When experimenting on Gaussian noise, choose "gauss5_50". 
# When experimenting on Poisson noise, choose "poisson5_50".

name="gauss5_50_try"  # Customize the folder name for saving the models

device='0'  #  GPU id


build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name gopro  --model selfir     --noisetype $type      --name $name        --dataroot $dataroot  \
    --patch_size 256      --niter 200        --lr_decay_iters 50    --save_imgs False   --lr 3e-4  \
    --batch_size 16       --print_freq 100   --calc_metrics True    --gpu_ids $device   --blur_loss 0 --crpp_patch 32  -j 4   | tee $LOG    


# If you want to utilize auxiliary loss to fine-tune the model, run the following code.

# finetune_name="gauss5_50_finetune_try"

# python train.py \
#     --dataset_name gopro  --model selfir     --noisetype $type      --name $finetune_name   --dataroot $dataroot  \
#     --patch_size 256      --niter 100        --lr_decay_iters 25    --save_imgs False       --lr 1e-4  \
#     --batch_size 16       --print_freq 100   --calc_metrics True    --gpu_ids $device       -j 4 \
#     --blur_loss 2         --crpp_patch 64    --load_path  ./ckpt/$name/UNET_model_200.pth   | tee $LOG 