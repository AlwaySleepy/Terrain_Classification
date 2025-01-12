from evaluate import evaluate
import os

data_type = "medium"  # select from "entry","easy","medium","hard"
test_datapath = "TCPOSS_data/h5py/test_medium.hdf5" # corresponding dataset (testset) path to data_type
model_type = "fft_res" # select from "mobilenet", "resnet", "densenet"
dir="fft_res_focalmore_medium"



if __name__ == "__main__":

    if os.path.exists("confusion_matrix/"+dir)==False:
        os.makedirs("confusion_matrix/"+dir)

    for i in range(1, 11):
        index=i*6
        model_path = f"model/{model_type}/{model_type}_{data_type}_focalmore_{index}.pth" # corresponding model path to data_type

        with open("evaluate_log.txt", "a") as f:
            f.write(f"model_path: {model_path}\n")

        evaluate(model_type, model_path, test_datapath, data_type,dir,index)
    
    # model_path = "model/fft_res/fft_res_hard_mix.pth" # corresponding model path to data_type