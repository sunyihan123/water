import os
from PIL import Image


def compress_images(folder_path, output_path, output_size):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 确保是图片文件
            # 打开图片并调整大小
            img = Image.open(os.path.join(folder_path, filename))
            img = img.resize(output_size, Image.ANTIALIAS)

            # 获取旧文件名的前八位
            base_filename = filename[:8]

            # 根据前六位创建子文件夹
            sub_folder = os.path.join(output_path, base_filename[:6])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            img = img.convert("RGB")
            # 保存新图片到子文件夹中，文件名为旧文件名的前八位
            img.save(os.path.join(sub_folder, base_filename + ".jpg"))

        # 调用函数在。


folder_path = r"D:\资料库\GAN\GAN-LSTM\my_data\zaolansu\原始数据\2010"
output_path = r"D:\资料库\GAN\GAN-LSTM\my_data\zaolansu\2010"
# 替换为您的文件夹路径
output_size = (512, 512)
compress_images(folder_path, output_path, output_size)



