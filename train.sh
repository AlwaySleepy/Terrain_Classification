#!/bin/bash

# Command for entry dataset
# python train.py --repeat_time 1 --model_type m_densenet --is_pretrained --is_augment --is_dropout --epochs 10 --is_save --train_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --test_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --save_path "./model/m_densenet_entry.pth"
# python train.py --repeat_time 1 --model_type m_densenet --is_pretrained --is_augment --is_dropout --epochs 10 --is_save --train_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --test_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --save_path "./model/m_densenet_easy.pth"
# python train.py --repeat_time 1 --model_type m_densenet --is_pretrained --is_augment --is_dropout --epochs 10 --is_save --train_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --test_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --save_path "./model/m_densenet_medium.pth"
# python train.py --repeat_time 1 --model_type m_densenet --is_pretrained --is_augment --is_dropout --epochs 10 --is_save --train_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --test_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --save_path "./model/m_densenet_hard.pth"


#!/bin/bash

# 创建保存模型的目录
mkdir -p model/mix_densenet

# Command for entry dataset
# python train.py --repeat_time 1 --model_type mix_densenet --is_pretrained --is_augment --is_dropout --epochs 30 --is_save --train_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --test_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --save_path "./model/mix_densenet/mix_densenet_entry.pth"
# python train.py --repeat_time 1 --model_type mix_densenet --is_pretrained --is_augment --is_dropout --epochs 30 --is_save --train_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --test_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --save_path "./model/mix_densenet/mix_densenet_easy.pth"
python train.py --repeat_time 1 --model_type mix_densenet --is_pretrained --is_augment --is_dropout --epochs 30 --is_save --train_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --test_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --save_path "./model/mix_densenet/mix_densenet_medium.pth"
python train.py --repeat_time 1 --model_type mix_densenet --is_pretrained --is_augment --is_dropout --epochs 30 --is_save --train_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --test_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --save_path "./model/mix_densenet/mix_densenet_hard.pth"
