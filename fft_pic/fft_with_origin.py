
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

def fourier_transform( img):
    # 假设 img 是一个 RGB 图像的 numpy 数组，形状为 (H, W, 3)
    print(img.shape)
    # img = img.transpose(1, 2, 0)  # 转换为 (224, 224, 3)

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
    cv2.circle(low_pass_mask, (ccol, crow), 20, 1, -1)  # 半径20的圆形

    # 高频掩模
    high_pass_mask = 1 - low_pass_mask

    # 应用掩模
    high_pass_dft = dft_shift * high_pass_mask

    # 反变换
    high_pass_img = np.fft.ifft2(np.fft.ifftshift(high_pass_dft)).real
    # print(high_pass_img.shape)
    high_pass_img=np.expand_dims(high_pass_img, axis=0)
    return high_pass_img
    


def process_and_multiply(fft_img, img,img_path):
    # Step 1: 归一化并反转灰度图像
    def normalize_and_invert(fft_img):
        # fft_img = fft_img.float()
        min_val = fft_img.min()
        max_val = fft_img.max()
        fft_img_inverted = max_val - fft_img
        normalized_img = (fft_img_inverted - min_val) / (max_val - min_val)
        return normalized_img

    # 归一化和反转
    normalized_img = normalize_and_invert(fft_img)

    # Step 2: 复制灰度图像成 3 通道
    # 假设输入 fft_img 是 [H, W]，复制后变成 [3, H, W]
    print(normalized_img.shape)
    normalized_img=torch.from_numpy(normalized_img)
    three_channel_img = normalized_img.repeat(3, 1, 1)  # 扩展维度并复制
    three_channel_img = three_channel_img.numpy().transpose(1, 2, 0)  # 转换为 (224, 224, 3)

    # Step 3: 逐元素与 RGB 图像相乘
    # 假设 img 是 [3, H, W]
    result = three_channel_img * img
    # result=result/255.0
    print(result.shape)
    result=result.astype(np.float32)
    # plt.clf()
    # plt.figure(figsize=(10, 8))
    # plt.imshow(result)
    # plt.savefig(f'fft_pic/mul/{img_path}.jpg')
    # 使用 OpenCV 显示图像
    cv2.imshow('Result', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 转换为 BGR 格式
    cv2.waitKey(0)  # 等待按键关闭窗口
    cv2.destroyAllWindows()  # 关闭所有窗口

    return result


if __name__ == '__main__':
    img = cv2.imread('fft_pic/fft/flagstone.jpg')

    fft_img = fourier_transform(img)
    process_and_multiply(fft_img, img,"flagstone")
