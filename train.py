import random
import torch
from torch import nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys
import datetime
import configparser
from config.config_train import config
import argparse
import time
from losses import FocalLoss, SupervisedContrastiveLoss


import torchvision

# Create a new folder to record parameters named with current time
folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir('./record/' + folder_name)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, type=int, default=40)
parser.add_argument('--batch_size', required=False, type=int, default=64)
parser.add_argument('--learning_rate', required=False, type=float, default=1e-3)
parser.add_argument('--lr_decay', required=False, type=float, default=0.9)
parser.add_argument('--n_classes', required=False, type=int, default=10)
parser.add_argument('--train_datapath', required=False, type=str, default="./TCPOSS_data/h5py/train_entry.hdf5")
parser.add_argument('--test_datapath', required=False, type=str, default="./TCPOSS_data/h5py/test_entry.hdf5")
parser.add_argument('--save_path', required=False, type=str, default='./model/imgmodel_entry.pth')
parser.add_argument('--what_we_want', required=False, type=str, default="Default")
parser.add_argument('--repeat_time', required=False, type=int, default=5)
parser.add_argument('--is_save', required=False, action="store_true")
parser.add_argument('--is_frozen', required=False, action="store_true")
parser.add_argument('--is_pretrained', required=False, action="store_true")
parser.add_argument('--is_augment', required=False, action="store_true")
parser.add_argument('--save_all', required=False, action="store_true")
parser.add_argument('--is_dropout', required=False, action="store_true")
parser.add_argument('--model_type', required=False, type=str, default="mobilenet", help="mobilenet, resnet, densenet")
args = parser.parse_known_args()[0]

# if no parameters are passed by command line parameters, use config parameters in ./config/config_train.py
print("sys.argv: ", sys.argv)
print("args: ", args)
if len(sys.argv) == 1:
    print("using config file:")
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    n_classes = config.n_classes
    train_datapath = config.train_datapath
    test_datapath = config.test_datapath
    save_path = config.save_path
    is_save = config.is_save
    is_frozen = config.is_frozen
    is_pretrained = config.is_pretrained
    is_augment = config.is_augment
    repeat_time = config.repeat_time
    lr_decay = config.lr_decay
    what_we_want = config.what_we_want
    save_all = config.save_all
    is_dropout = config.is_dropout
    model_type = config.model_type
else:
    print("using command line args:")
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    n_classes = args.n_classes
    train_datapath = args.train_datapath
    test_datapath = args.test_datapath
    save_path = args.save_path
    is_save = args.is_save
    is_frozen = args.is_frozen
    is_pretrained = args.is_pretrained
    is_augment = args.is_augment
    repeat_time = args.repeat_time
    lr_decay = args.lr_decay
    what_we_want = args.what_we_want
    save_all = args.save_all
    is_dropout = args.is_dropout
    model_type = args.model_type

with open ('./record/' + folder_name + '/config.txt', 'w') as f:
    f.write(str(what_we_want) + "\n")
    f.write("batch_size: " + str(batch_size) + "\n")
    f.write("learning_rate: " + str(learning_rate) + "\n")
    f.write("epochs: " + str(epochs) + "\n")
    f.write("n_classes: " + str(n_classes) + "\n")
    f.write("train_datapath: " + str(train_datapath) + "\n")
    f.write("test_datapath: " + str(test_datapath) + "\n")
    f.write("is_save: " + str(is_save) + "\n")
    f.write("save_path: " + str(save_path) + "\n")
    f.write("is_frozen: " + str(is_frozen) + "\n")
    f.write("is_pretrained: " + str(is_pretrained) + "\n")
    f.write("is_augment: " + str(is_augment) + "\n")
    f.write("repeat_time: " + str(repeat_time) + "\n")
    f.write("lr_decay: "+ str(lr_decay) + "\n")
    f.write("save_all: " + str(save_all) + "\n")
    f.write("is_dropout: " + str(is_dropout) + "\n")
    f.write("model_type: " + str(model_type) + "\n")

def load_data(datapath, mode='train'):
    f = h5py.File(datapath, 'r')
    labels = f['labels']['labels'][:]
    images = f['images']['images'][:]
    print(images.shape,"# of images")
    print(labels.shape,"# of labels")
    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    if (mode == 'train') & is_augment:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # normilize helps a little
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # normilize helps a little
        ])
    data = ImageDataset(labels, images, transform=transform)
    print(len(data), "# data length")
    return data
model_name = {"mobilenet": "mobilenet_v2", "resnet": "resnet50", "densenet": "densenet121","fft_res":"fft_resnet"}
feature_dim = {"mobilenet": 1280, "resnet": 2048, "densenet": 1024,"fft_res":512}
start_time = time.time()
print("trainset:")
training_data = load_data(train_datapath)
test_data = load_data(test_datapath, mode='test')
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
labelp = ['asphalt','grass','cement','board','brick','gravel','sand','flagstone','plastic','soil']
total_train_accs = []
total_test_accs = []
if repeat_time >= 1:
    for i in range(repeat_time):
        print("repeat_time: ", i+1)

        if model_name[model_type] =="fft_resnet":
            from model.fft_resnet import resnet50
            model = resnet50(device="cuda:0",classes=10,network='resnet50')
        else:
            # build model
            if is_pretrained:
                # model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_type], pretrained=True)
                model=torchvision.models.densenet121(pretrained=True)
            else:
                model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_type], pretrained=False)


        # for child in model.named_children():
        #     print(child)
        if is_dropout:
            # print(feature_dim[model_type])
            classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(feature_dim[model_type], 32),
                nn.Dropout(0.25),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )
        else:
            classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(feature_dim[model_type], 32),
                nn.Linear(32, n_classes),
            )

        if model_name[model_type] == "fft_resnet":
            pass
        else:
            model.classifier = classifier
        # only finetune classifier
        if is_frozen:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        # gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # train and validate
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        loss_fn = nn.CrossEntropyLoss()

        alpha=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        loss_focal=FocalLoss(alpha=alpha)
        loss_contrastive = SupervisedContrastiveLoss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # learning rate decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)
        for epoch in range(epochs):
            model.train()
            loss_list = []
            total_correct = 0
            total = 0
            pbar = tqdm(total=len(train_dataloader))
            for batch, (X, y) in enumerate(train_dataloader):
                # if model_name[model_type] == "fft_densenet" or model_name[model_type] == "modified_densenet":
                #     # fft_imgs = torch.tensor([model.fourier_transform(x) for x in X.numpy()]).to(device)
                #     fft_imgs_list = [model.fourier_transform(x) for x in X.numpy()]
                #     # 将列表转为单个 NumPy 数组
                #     fft_imgs_np = np.array(fft_imgs_list)

                #     # 再将 NumPy 数组转换为 PyTorch 张量
                #     fft_imgs = torch.from_numpy(fft_imgs_np).to(device)

                imgs = X.to(device)
                y = (y).type(torch.LongTensor).to(device)

                if model_name[model_type] == "fft_resnet":
                    aug_list = ['spatial_dropout', 'freq_dropout', 'freq_noise', 'freq_mixup']
                    # p = ((epoch+1) / epochs) ** 0.5 * 0.8
                    # if p<0.5:
                    #     p=0.5
                    y_hat= model(imgs, labels=y, aug_mode=random.choice(aug_list))
                    features=model.get_features(imgs)
                else:
                    y_hat = model(imgs)

                loss = loss_focal(y_hat, y)
                if model_name[model_type] == "fft_resnet":
                    loss_contrast = loss_contrastive(features, y)
                    # print(loss_contrast)
                    loss = 10*loss + 0.2*loss_contrast
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
                total_correct += correct
                total += len(y)
                loss_list.append(loss.item())
                pbar.set_description(f"epoch {epoch+1}")
                pbar.set_postfix(loss=f"{loss.item():>7f}", accuracy=f"{total_correct/total:>7f}")
                pbar.update(1)

                # del fft_imgs
                # torch.cuda.empty_cache()

            pbar.close()
            print(f"In epoch {epoch+1}, total train accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
            train_losses.append(np.mean(loss_list))
            train_accs.append((total_correct/total).item())
            scheduler.step()

            torch.cuda.empty_cache()
            model.eval()
            pbar = tqdm(total = len(test_dataloader))
            loss_list = []
            total_correct = 0
            total = 0
            with torch.no_grad():
                for batch, (X, y) in enumerate(test_dataloader):
                    # if model_name[model_type] == "fft_densenet" or model_name[model_type] == "modified_densenet":
                    #     # fft_imgs = torch.tensor([model.fourier_transform(x) for x in X.numpy()]).to(device)
                    #     fft_imgs_list = [model.fourier_transform(x) for x in X.numpy()]
                    #     # 将列表转为单个 NumPy 数组
                    #     fft_imgs_np = np.array(fft_imgs_list)

                    #     # 再将 NumPy 数组转换为 PyTorch 张量
                    #     fft_imgs = torch.from_numpy(fft_imgs_np).to(device)

                    imgs = X.to(device)
                    y = (y).type(torch.LongTensor).to(device)
                    # if model_name[model_type] == "fft_densenet":
                    #     y_hat = model(imgs, fft_imgs)
                    # elif model_name[model_type] == "modified_densenet":
                    #     y_hat = model(fft_imgs)
                    # elif model_name[model_type] == "fft_resnet":
                    #     y_hat = model(imgs, aug_mode='freq_dropout')
             
                    y_hat = model(imgs)
                    # imgs = X.to(device)
                    # y_hat = model(imgs)
                    y = (y).type(torch.LongTensor).to(device)
                    loss = loss_focal(y_hat, y)
                    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
                    total_correct += correct
                    total += len(y)
                    loss_list.append(loss.item())
                    pbar.set_description(f"epoch {epoch+1}")
                    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
                    pbar.update(1)
                    # del fft_imgs
                    # torch.cuda.empty_cache()

            pbar.close()
            print(f"In epoch {epoch+1}, total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
            test_losses.append(np.mean(loss_list))
            test_accs.append((total_correct/total).item())
            if (epoch+1)%6==0:
                if is_save:
                    torch.save(model.state_dict(), save_path[:-4]+ "_" + str(epoch+1) + ".pth")
                    print("save successfully")
        total_train_accs.append(train_accs[-1])
        total_test_accs.append(test_accs[-1])

end_time = time.time()
mean_train_acc = np.mean(total_train_accs)
var_train_acc = np.var(total_train_accs)
mean_test_acc = np.mean(total_test_accs)
var_test_acc = np.var(total_test_accs)
range_train_acc = np.max(total_train_accs) - np.min(total_train_accs)
half_range_train_acc = max(np.max(total_train_accs) - mean_train_acc, mean_train_acc - np.min(total_train_accs))
range_test_acc = np.max(total_test_accs) - np.min(total_test_accs)
half_range_test_acc = max(np.max(total_test_accs) - mean_test_acc, mean_test_acc - np.min(total_test_accs))
print("mean_train_acc: ", mean_train_acc, " +- ", half_range_train_acc)
print("range_train_acc: ", range_train_acc)
print("var_train_acc: ", var_train_acc)
print("mean_test_acc: ", mean_test_acc, " +- ", half_range_test_acc)
print("range_test_acc: ", range_test_acc)
print("var_test_acc: ", var_test_acc)

# record loss and accuracy (last for repreat)
with open('./record/' + folder_name + '/loss_acc.txt', 'w') as f:
    f.write("train_losses:\n")
    for i in train_losses:
        f.write(str(i) + ", ")
    f.write("\ntrain_accs:\n")
    for i in train_accs:
        f.write(str(i) + ", ")
    f.write("\ntest_losses:\n")
    for i in test_losses:
        f.write(str(i) + ", ")
    f.write("\ntest_accs:\n")
    for i in test_accs:
        f.write(str(i) + ", ")
    f.write("\nmean_train_acc: " + str(mean_train_acc) + " +- " + str(half_range_train_acc) + "\n")
    f.write("var_train_acc: " + str(var_train_acc) + "\n")
    f.write("mean_test_acc: " + str(mean_test_acc) + " +- " + str(half_range_test_acc) + "\n")
    f.write("var_test_acc: " + str(var_test_acc) + "\n")
    f.write("total_train_accs:\n")
    for i in total_train_accs:
        f.write(str(i) + ", ")
    f.write("\ntotal_test_accs:\n")
    for i in total_test_accs:
        f.write(str(i) + ", ")
    f.write("total process time: " + str(end_time - start_time) + "seconds\n")

# plot loss and accuracy
epoch_x = np.arange(epochs)+1
plt.figure()
plt.plot(epoch_x, train_losses, label='train loss')
plt.plot(epoch_x, test_losses, label='test loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./record/' + folder_name + '/loss.png')
plt.figure()
plt.plot(epoch_x, train_accs, label='train accuracy')
plt.plot(epoch_x, test_accs, label='test accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('./record/' + folder_name + '/accuracy.png')

# save model
if is_save & (not save_all):
    torch.save(model.state_dict(), save_path)