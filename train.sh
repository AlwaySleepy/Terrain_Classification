#!/bin/bash


mkdir -p model/fft_res


python train.py --repeat_time 1 --model_type fft_res --is_pretrained --is_augment --is_dropout --epochs 60 --is_save --train_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --test_datapath "./TCPOSS_data/h5py/train_entry.hdf5" --save_path "./model/fft_res/fft_res_entry_focalmore.pth"
python train.py --repeat_time 1 --model_type fft_res --is_pretrained --is_augment --is_dropout --epochs 60 --is_save --train_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --test_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --save_path "./model/fft_res/fft_res_easy_focalmore.pth"
python train.py --repeat_time 1 --model_type fft_res --is_pretrained --is_augment --is_dropout --epochs 60 --is_save --train_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --test_datapath "./TCPOSS_data/h5py/train_medium.hdf5" --save_path "./model/fft_res/fft_res_medium_focalmore.pth"
python train.py --repeat_time 1 --model_type fft_res --is_pretrained --is_augment --is_dropout --epochs 60 --is_save --train_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --test_datapath "./TCPOSS_data/h5py/train_hard.hdf5" --save_path "./model/fft_res/fft_res_hard_focalmore.pth"
