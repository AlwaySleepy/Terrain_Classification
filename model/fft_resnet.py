import torch
import random
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.fft as fft
import torchvision.models as models
import cv2

class ResNet(nn.Module):
    def __init__(self, block, layers,
                 device,
                 classes=100,
                 domains=2,
                 network='resnet18',
                 ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if network == "resnet18":
            layer_channels = [64, 128, 256, 512]
        else:
            layer_channels = [256, 512, 1024, 2048]

        self.device = device
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.n_domains = domains
        self.n_classes = classes
        self.n_channels = 512 * block.expansion 
        self.classifier = nn.Linear(self.n_channels, self.n_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def pooling(self, x):
        return self.avgpool(x).view(x.size(0), -1)
    

    def fourier_transform(self, img):
        # 假设 img 是一个 RGB 图像的 numpy 数组，形状为 (H, W, 3)
        img = img.transpose(1, 2, 0)  # 转换为 (224, 224, 3)

        # print(img.shape)
        x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 傅里叶变换
        dft = np.fft.fft2(x)
        dft_shift = np.fft.fftshift(dft)

        # 创建掩模
        rows, cols = x.shape
        crow, ccol = rows // 2, cols // 2  # 中心点

        # 低频掩模
        low_pass_mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(low_pass_mask, (ccol, crow), 10, 1, -1)  # 半径20的圆形

        # 高频掩模
        high_pass_mask = 1 - low_pass_mask

        # 应用掩模
        high_pass_dft = dft_shift * high_pass_mask

        # 反变换
        high_pass_img = np.fft.ifft2(np.fft.ifftshift(high_pass_dft)).real
        # print(high_pass_img.shape)
        high_pass_img=np.expand_dims(high_pass_img, axis=0)
        return high_pass_img


    def process_and_multiply(self, fft_img, img):
        # Step 1: 归一化并反转灰度图像
        def normalize_and_invert(fft_img):
            # print(fft_img)
            fft_img = fft_img.float()
            min_val = fft_img.min()
            max_val = fft_img.max()
            # print(min_val,max_val)

            normalized_img = (fft_img - min_val) / (max_val - min_val)
            normalized_img=1.0-normalized_img
            # max_normalized_img=normalized_img.max()
            # min_normalized_img=normalized_img.min()
            # print(max_normalized_img,min_normalized_img)
            # normalized_img=normalized_img ** 2
            mask=normalized_img>0.6
            normalized_img[mask]=1
            normalized_img=normalized_img ** 2
            return normalized_img

        # 归一化和反转
        normalized_img = normalize_and_invert(fft_img)
        # print(normalized_img)
        # Step 2: 复制灰度图像成 3 通道
        three_channel_img = normalized_img.repeat(1, 3, 1, 1)  # 扩展维度并复制

    
        # Step 3: 混合图像和白色背景
        white = torch.ones_like(img)  # 创建一个全白的背景
        result = (1 - three_channel_img) * white + three_channel_img * img
        return result


    def forward(self, x, labels=None, aug_mode=False):

        # # if True:
        # # print("trick in")
        # fft_img=[self.fourier_transform(i) for i in x.cpu().numpy()]
        # fft_imgs_np = np.array(fft_img)
        # # 再将 NumPy 数组转换为 PyTorch 张量
        # fft_imgs = torch.from_numpy(fft_imgs_np).to("cuda:0")
        # x=self.process_and_multiply(fft_imgs, x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        aug_idx = random.choice([0,1,2,3]) 
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]        
        for idx, layer in enumerate(layers):
            x = layer(x)
            if aug_idx==idx and aug_mode=='freq_noise':
                x = self.magnitude_noise(x, std=0.5)
            if aug_idx==idx and aug_mode=='freq_dropout':
                x = self.magnitude_dropout(x, p=0.5)
            if aug_idx==idx and aug_mode=='freq_mixup':
                x = self.magnitude_mixup(x) 
            if aug_idx==idx and aug_mode=='spatial_dropout':
                res_layers = [layers[i] for i in range(idx+1, len(layers))]
                x = self.spatial_dropout(res_layers, x, labels) 
        # print(x.shape)
        logit = self.classifier(self.pooling(x)) 
        return logit

   

    def get_features(self, x, magnitude_layer=False):
        x = self.conv1(x)
        x = self.bn1(x)
        if magnitude_layer == 'conv1':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if magnitude_layer == 'layer1':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer2(x)
        if magnitude_layer == 'layer2':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer3(x)
        if magnitude_layer == 'layer3':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer4(x)
        if magnitude_layer == 'layer4':
            _, magnitude = self.fft(x, is_shift=True)

        if magnitude_layer:
            return self.pooling(x), magnitude.mean(dim=1)
        else:
            return self.pooling(x)     

    def spatial_dropout(self, res_layers, x, labels, p=0.6):
        # ger final features
        x_bar = x.clone().detach()
        if len(res_layers) > 0:
            with torch.no_grad():
                for idx, layer in enumerate(res_layers):
                    x_bar = layer(x_bar)

        # generate attention map
        fc_weights = self.classifier.weight.data  
        conv_weights = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1)  
        logit = F.conv2d(x_bar, conv_weights)  
        b, c, h, w = logit.size()

        norm_logit = torch.zeros((b, h, w)).to(self.device)
        # for i in range(labels.size(0)):
        #     norm_attn[i] = (-probabilities[i, labels[i]] * torch.log2(probabilities[i, labels[i]] + 1e-12))
        for i in range(labels.size(0)):
            norm_logit[i] = logit[i, labels[i]]
        norm_logit = norm_logit.view(b, h*w)
        logit_max  = norm_logit.max(dim=-1)[0].unsqueeze(dim=-1)
        logit_min  = norm_logit.min(dim=-1)[0].unsqueeze(dim=-1)
        norm_logit = (norm_logit - logit_min) / (logit_max - logit_min)
        norm_logit = norm_logit.view(b, h, w).unsqueeze(dim=1)

        norm_ent = torch.zeros((b, h, w)).to(self.device)
        probabilities = F.softmax(logit, dim=1)
        for i in range(labels.size(0)):
            norm_ent[i] = (-probabilities[i, labels[i]] * torch.log2(probabilities[i, labels[i]] + 1e-12))
        norm_ent = norm_ent.view(b, h*w)
        ent_max  = norm_ent.max(dim=-1)[0].unsqueeze(dim=-1)
        ent_min  = norm_ent.min(dim=-1)[0].unsqueeze(dim=-1)
        norm_ent = 1 - (norm_ent - ent_min) / (ent_max - ent_min)
        norm_ent = norm_ent.view(b, h, w).unsqueeze(dim=1)

        norm_attn = (norm_ent - norm_logit).clip(min=0)
        attn_max  = norm_attn.max(dim=-1)[0].unsqueeze(dim=-1)
        attn_min  = norm_attn.min(dim=-1)[0].unsqueeze(dim=-1)
        norm_attn = 1 - (norm_attn - attn_min) / (attn_max - attn_min)
        norm_attn = norm_attn.view(b, h, w).unsqueeze(dim=1)
        # mask = F.interpolate(mask, size=(7,7), mode='bilinear')

        # generate mask
        mask = torch.rand(norm_logit.size()).to(self.device).detach() * norm_logit
        mask = (mask < (1-p)).float()
        mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear')
        return x*mask

    def fft(self, x, is_shift=False):
        spectrum = fft.fft2(x, dim=(-2, -1))  # 在空间维度上执行2D傅里叶变换
        phase = torch.angle(spectrum)
        magnitude = torch.abs(spectrum)
        if is_shift:
            magnitude = fft.ifftshift(magnitude)
        return phase, magnitude     

    def ifft(self, magnitude, phase):
        reconstructed_spectrum = magnitude * torch.exp(1j * phase)
        reconstructed_x = fft.ifft2(reconstructed_spectrum, dim=(-2, -1)).real
        return reconstructed_x   


    def magnitude_noise(self, x, std=0.5):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: noising
        batch_size, channels, height, width = x.size()
        white_noise = torch.randn((batch_size, 1, height, width)).to(self.device).detach()
        scaled_white_noise = (1 + std * white_noise).clip(min=0)
        noised_magnitude = scaled_white_noise*magnitude

        # reconstruct images
        reconstructed_x = self.ifft(noised_magnitude, phase)
        return reconstructed_x

    def magnitude_dropout(self, x, p=0.5):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: dropout
        batch_size, _, height, width = x.size()
        mask = torch.rand((batch_size, 1, height, width)).to(self.device)
        mask = (mask > p).float().detach()
        mask[:,:,0,0] = 1
        dropped_magnitude = mask*magnitude

        # reconstruct images
        reconstructed_x = self.ifft(dropped_magnitude, phase)
        return reconstructed_x

    def magnitude_mixup(self, x):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: magnitude mixup
        batch_size = x.size(0)
        lam = torch.rand(batch_size).to(self.device).detach()\
            .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        index = torch.randperm(batch_size)
        mixed_magnitude = lam * magnitude + (1-lam) * magnitude[index]

        # reconstruct images
        reconstructed_x = self.ifft(mixed_magnitude, phase)
        return reconstructed_x





def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # Use torchvision.models.resnet18() to load pre-trained weights
        pretrained_model = models.resnet18(pretrained=True)
        model_dict = model.state_dict()

        # Get the state dict of the pretrained model (ResNet-18)
        pretrained_dict = pretrained_model.state_dict()

        # Filter out any keys that are not in the model's state_dict (e.g., classifier layer)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update the model's state_dict with the pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated state dict into the model
        model.load_state_dict(model_dict, strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # Use torchvision.models.resnet50() to load pre-trained weights
        pretrained_model = models.resnet50(pretrained=True)
        model_dict = model.state_dict()

        # Get the state dict of the pretrained model (ResNet-50)
        pretrained_dict = pretrained_model.state_dict()

        # Filter out any keys that are not in the model's state_dict (e.g., classifier layer)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update the model's state_dict with the pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated state dict into the model
        model.load_state_dict(model_dict, strict=False)
    return model