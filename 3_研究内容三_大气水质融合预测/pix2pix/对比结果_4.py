import cv2
import numpy as np

def compare_images_histogram(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 将图片转换为HSV颜色空间
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # 计算图片的直方图
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 归一化直方图
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX, -1)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX, -1)

    # 计算直方图的差异
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity

# 示例用法


img1_path = 'image1.jpg'
img2_path = 'image2.jpg'
similarity = compare_images_histogram(img1_path, img2_path)
print('相似度：', similarity)
