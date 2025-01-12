import torch
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import time
import pandas



data_type = "entry"  # select from "entry","easy","medium","hard"
test_datapath = "TCPOSS_data/h5py/test_entry.hdf5" # corresponding dataset (testset) path to data_type
model_type = "fft_res" # select from "mobilenet", "resnet", "densenet"
model_path = "model/fft_res/fft_res_entry_focalmore.pth" # corresponding model path to data_type


def evaluate(model_type, model_path, test_datapath, data_type,dir,index):
    model_path = model_path
    model_type = model_type
    test_datapath = test_datapath
    data_type = data_type

    #hyperparameters
    batch_size = 64
    n_classes = 10
    feature_dim = {"mobilenet": 1280, "resnet": 2048, "densenet": 1024,"fft_res":512}
    model_name = {"mobilenet": "mobilenet_v2", "resnet": "resnet50", "densenet": "densenet121","fft_res":"fft_resnet"}
    data = {'random':np.zeros((5,)), 'easy':np.zeros((5,)), 'medium':np.zeros((5,)), 'hard':np.zeros((5,))}
    classes = ['asphalt', 'grass', 'cement', 'board', 'brick', 'gravel', 'sand', 'flagstone', 'plastic', 'soil']
    modes = ['random', 'easy', 'medium', 'hard']
    # model architecture

    if model_name[model_type] == "fft_resnet":
        from model.fft_resnet import resnet50
        model = resnet50(device="cuda:0",classes=10)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_type])
    classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(feature_dim[model_type], 32),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )
    if model_type == 'resnet' or model_type == 'mix_resnet':
        model.fc = classifier
    elif model_name[model_type] == "fft_densenet" or model_name[model_type] == "fft_resnet":
        pass
    else:
        model.classifier = classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    model.load_state_dict(torch.load(model_path))
    model.eval()
    def load_data(datapath):
        f = h5py.File(datapath, 'r')
        labels = f['labels']['labels'][:]
        images = f['images']['images'][:]
        print(images.shape,"# of images")
        print(labels.shape,"# of labels")
        f.close()
        images = images.reshape(images.shape[0], 224, 224, 3)/255.0
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # normilize helps a little
        ])
        data = ImageDataset(labels, images, transform=transform)
        return data
    print("testset:")
    test_data = load_data(test_datapath)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    # start evaluation
    gt = np.empty((0,),dtype=np.int32)
    pd = np.empty((0,),dtype=np.int32)
    probs = np.empty((0,n_classes),dtype=np.float32)
    total_correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(total=len(test_dataloader))
        for X,y in test_dataloader:
            # if model_name[model_type] == "fft_densenet" or model_name[model_type] == "modified_densenet":
            #     # fft_imgs = torch.tensor([model.fourier_transform(x) for x in X.numpy()]).to(device)
            #     fft_imgs_list = [model.fourier_transform(x) for x in X.numpy()]
            #     # 将列表转为单个 NumPy 数组
            #     fft_imgs_np = np.array(fft_imgs_list)

            #     # 再将 NumPy 数组转换为 PyTorch 张量
            #     fft_imgs = torch.from_numpy(fft_imgs_np).to(device)

            imgs = X.to(device)
            label = y
            # if model_name[model_type] == "fft_densenet":
            #     yhat = model(imgs, fft_imgs)
            # elif model_name[model_type] == "modified_densenet":
            #     yhat = model(fft_imgs)

            yhat = model(imgs)
            
            pred = torch.argmax(yhat, dim=1)
            prob = F.softmax(yhat, dim=1)
            gt = np.append(gt, label.numpy())
            pd = np.append(pd, pred.cpu().numpy())
            correct = np.sum(pred.cpu().numpy() == y.numpy())
            total_correct += correct
            total += len(y)
            probs = np.append(probs, prob.cpu().numpy(), axis=0)
            pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
            pbar.update(1)
        pbar.close()
    coding=n_classes*gt+pd  # Utilize unique encoding.
    print(gt)
    print(pd)
    confusion_matrix_1d=np.bincount(coding)
    confusion_matrix=confusion_matrix_1d.reshape(n_classes,n_classes)  # confusion matrix
    print("Category matrix (testset):", confusion_matrix)
    # Calculate recall and precision for each category.
    recall = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    print("Data size (testset):",np.sum(confusion_matrix,axis=1))
    for i in range(n_classes):
        recall[i] = confusion_matrix[i,i]/np.sum(confusion_matrix,axis=1)[i]
        precision[i] = confusion_matrix[i,i]/np.sum(confusion_matrix,axis=0)[i]
    print("Recall of each category (divide every row):", recall)
    print("Precision of each category (divide every column):", precision)
    for classi in range(n_classes):
        with open("evaluate_log.txt", "a") as f:
            f.write(f"class:  {classes[classi]}  precision:  {precision[classi]}\n")
        print("class: ", classes[classi], ", total samples: ",np.sum(confusion_matrix,axis=1)[classi],"  recall: ",recall[classi] , "  precision: ", precision[classi])
    total_acc = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
    with open("evaluate_log.txt", "a") as f:
        f.write(f"total accurracy is: {total_acc}\n")
    print("total accurracy is: ",total_acc)
    end_time = time.time()
    print("time cost: ", end_time - start_time)
    # Save confusion_matrix as .csv file
    import pandas 
    df = pandas.DataFrame(confusion_matrix)
    df.columns = classes
    df.index = classes
    df.to_csv(f"./confusion_matrix/{dir}/{data_type}_{index}_confusion_matrix.csv")
    recall =pandas.Series(recall)
    recall.index = classes
    recall.name = 'recall'
    recall.to_csv(f"./confusion_matrix/{dir}/{data_type}_{index}_recall.csv")
    with open("evaluate_log.txt", "a") as f:
        f.write("\n")


if __name__ == "__main__":
    evaluate(model_type, model_path, test_datapath, data_type,"fft_res_focalmore_hard",101)
 