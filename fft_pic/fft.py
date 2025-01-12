import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_fft(img_path):

    img = cv2.imread(f'fft_pic/origin/{img_path}.tif', 0)
    print(img)
    #img = cv2.imread('Fig0421.tif', 0)
    dft = np.abs(np.fft.fft2(img))
    log_dft = np.log(1+dft)
    center_dft = np.fft.fftshift(log_dft)

    img_list = [img, dft, log_dft, center_dft]
    img_name_list = ['original', 'DFT', 'log transformed DFT', 'centralized DFT']

    _, axs = plt.subplots(2, 2)

    for i in range(2):
        for j in range(2):
            axs[i, j].imshow(img_list[i*2+j], cmap='gray')
            axs[i, j].set_title(img_name_list[i*2+j])
            axs[i, j].axis('off')

    plt.savefig(f'fft_pic/fft/{img_path}.jpg')
    plt.show()


def low_hight_filter(img_path):
    # 读取图像
    img = cv2.imread(f'fft_pic/origin/{img_path}.tif', 0)

    # 傅里叶变换
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # 创建掩模
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # 中心点

    # 低频掩模
    low_pass_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(low_pass_mask, (ccol, crow), 10, 1, -1)  # 半径30的圆形

    # 高频掩模
    high_pass_mask = 1 - low_pass_mask

    # 应用掩模
    low_pass_dft = dft_shift * low_pass_mask
    high_pass_dft = dft_shift * high_pass_mask

    # 反变换
    low_pass_img = np.fft.ifft2(np.fft.ifftshift(low_pass_dft)).real
    high_pass_img = np.fft.ifft2(np.fft.ifftshift(high_pass_dft)).real

    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(low_pass_img, cmap='gray'), plt.title('Low Pass Filtered')
    plt.subplot(133), plt.imshow(high_pass_img, cmap='gray'), plt.title('High Pass Filtered')
    plt.savefig(f'fft_pic/filter/{img_path}.jpg')
    plt.show()

if __name__ == '__main__':
    show_fft('grav')
    low_hight_filter('grav')