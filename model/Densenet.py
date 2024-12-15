import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class ModifiedDenseNet121(nn.Module):
    def __init__(self, original_densenet):
        super(ModifiedDenseNet121, self).__init__()
        self.densenet = original_densenet
        self.auxiliary_conv = nn.Conv2d(1024, 10, kernel_size=1)  # 1x1卷积作为辅助分类器
        self.threshold = 0.1  # 梯度抑制阈值
        self.classifier=None

    def forward(self, x):
        # 通过DenseNet的主体
        # print(x.shape)
        features = self.densenet.features(x)
        # print(features.shape)
        # 获取DenseNet最后的特征图

        # 计算辅助分类器的输出
        auxiliary_output = self.auxiliary_conv(features)
        
        # 计算输出的损失，这里假设使用交叉熵损失
        # print(auxiliary_output.shape)
        # print(labels.shape)
        # auxiliary_loss = F.cross_entropy(auxiliary_output.view(auxiliary_output.size(0), -1), labels)

        # # 计算梯度并屏蔽大的梯度
        # gradients = torch.abs(features.grad)  # 计算梯度绝对值
        # mask = gradients > self.threshold  # 标记过大的梯度
        # features[mask] = 0  # 将过大梯度对应的特征图屏蔽掉

        # # 最终输出
        # final_output = self.densenet.classifier(features)
        flatten_output=auxiliary_output.view(auxiliary_output.size(0), -1)
        # print(self.classifier)
        final_output=self.classifier(flatten_output)
        
        return final_output

# 加载原始的DenseNet121
original_densenet = torchvision.models.densenet121(pretrained=True)
modified_densenet = ModifiedDenseNet121(original_densenet)