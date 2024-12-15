import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from model.mixstyle import MixStyle

class Mix_Densenet(nn.Module):
    def __init__(self, original_densenet):
        super(Mix_Densenet, self).__init__()
        self.densenet = original_densenet
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.auxiliary_conv = nn.Conv2d(1024, 10, kernel_size=1)  # 1x1卷积作为辅助分类器


    def forward(self, x):
        # DenseNet 的前几个层
        # print(self.densenet)
        x = self.densenet.features[0](x)
        x = self.densenet.features[1](x)
        x = self.densenet.features[2](x)
        x = self.densenet.features[3](x)


        # 第一密集块
        x = self.densenet.features[4](x)
        x = self.mixstyle(x)  # 在第一块后应用 MixStyle

        x=self.densenet.features[5](x)
        # 第二密集块
        x = self.densenet.features[6](x)
        x = self.mixstyle(x)  # 在第二块后应用 MixStyle

        x=self.densenet.features[7](x)

        # 第三密集块
        x = self.densenet.features[8](x)

        # 继续应用 DenseNet 后续层
        x = self.densenet.features[9](x)
        x = self.densenet.features[10](x)

        x = self.densenet.features[11](x)

        auxiliary_output = self.auxiliary_conv(x)
        
        flatten_output=auxiliary_output.view(auxiliary_output.size(0), -1)
        # print(self.classifier)
        final_output=self.classifier(flatten_output)
        return final_output
    
original_densenet = torchvision.models.densenet121(pretrained=True)
mix_densenet = Mix_Densenet(original_densenet)