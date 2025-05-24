#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd
import time
import numpy as np
import tensorflow as tf
import os
import glob
from matplotlib import pyplot as plt


time_start = time.time()

#
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
# # config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 程序按需申请内存
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存
# sess = tf.compat.v1.Session(config=config)

# imgs_path = glob.glob(r'C:\Users\vipuser\Desktop\pix2pix原始\train\0\*.jpg')
# imgs_path_T = glob.glob(r'C:\Users\vipuser\Desktop\pix2pix原始\test\0\*.jpg')
# imgs_path = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\train\0\*.png')
# imgs_path_T = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\test\0\*.png')

imgs_path = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\train\012\*.png')    # 训练集的三要素
imgs_path1 = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\train\0\*.png')     # 训练集的一要素
imgs_path_T = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\test\012\*.png')   # 测试集的三要素
imgs_path_T1 = glob.glob(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\3DGAN\test\0\*.png')    # 测试集的一要素


imgs_path_T_G = imgs_path_T1[1:len(imgs_path_T1)-60]    # 测试集目标
imgs_path_T_I = imgs_path_T[0:len(imgs_path_T)-3-180]    # 测试集输入
imgs_path_G = imgs_path1[1:len(imgs_path1)-400]          # 训练集目标  左闭右开
imgs_path_I = imgs_path[0:len(imgs_path)-3-1200]          # 训练集输入


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def normal(image):
    image = tf.cast(image, tf.float32)/127.5 - 1
    return image


def load_image(image_path):
    image = read_jpg(image_path)
    image = tf.image.resize(image, (256, 256))
    image = normal(image)   
    return image


# 获得训练集目标图片
image_G = []
for i in range(len(imgs_path_G)):
    image_G.append(load_image(imgs_path_G[i]))


# 获得测试集目标图片
image_T_G = []
for i in range(len(imgs_path_T_G)):
    image_T_G.append(load_image(imgs_path_T_G[i]))


# 获得训练集输入图片
image_I = []
for i in range(len(imgs_path_I)):
    image_I.append(load_image(imgs_path_I[i]))


# 获得测试集输入图片
image_T_I = []
for i in range(len(imgs_path_T_I)):
    image_T_I.append(load_image(imgs_path_T_I[i]))


input_I = []    # 训练集 按时间顺序输入，12张为一组图片
for i in range(2, len(imgs_path_I), 3):
    a = np.concatenate((image_I[i-2], image_I[i-1], image_I[i]), axis=2)
    input_I.append(a)
print(len(input_I))


input_T_I = []  # 测试集 按时间顺序输入，12张为一组图片
for i in range(2, len(imgs_path_T_I), 3):
    a = np.concatenate((image_I[i - 2], image_I[i - 1], image_I[i]),axis=2)
    input_T_I.append(a)

#        训练集输入             训练集输出            测试集输入            测试集输出
print(np.shape(input_I), np.shape(image_G), np.shape(input_T_I), np.shape(image_T_G))

BATCH_SIZE = 1
auto = tf.data.experimental.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices((input_I, image_G))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)#训练集


dataset_T = tf.data.Dataset.from_tensor_slices((input_T_I, image_T_G))
dataset_T = dataset_T.batch(BATCH_SIZE)
dataset_T = dataset_T.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# plt.subplot(1, 2, 1)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(image_I[6]))
#
# plt.subplot(1, 2, 2)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(image_G[6]))
# plt.show()
#
#
# plt.subplot(1, 2, 1)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(image_T_I[6]))
#
# plt.subplot(1, 2, 2)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(image_T_G[6]))
# plt.show()


def down(filters, size, apply_bn=True):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False)
    )
    if apply_bn:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model


def up(filters, size, apply_drop=False):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, 
                                        padding='same', use_bias=False)
    )
    model.add(tf.keras.layers.BatchNormalization())
    if apply_drop:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.ReLU())
    return model


def Generator():
    inputs = tf.keras.layers.Input(shape=(256, 256, 9))
    
    down_stack = [
        down(64, 4, apply_bn=False),     # 128*128*64
        down(128, 4),                    # 64*64*128
        down(256, 4),                    # 32*32*256
        down(512, 4),                    # 16*16*512
        down(512, 4),                    # 8*8*512
        down(512, 4),                    # 4*4*512
        down(512, 4),                    # 2*2*512
        down(512, 4),                    # 1*1*512
        # down(512, 4),                    #1*1*512
        # down(512, 4),                    #1*1*512
    ]
    
    up_stack = [
        up(512, 4, apply_drop=True),      # 2*2*512     drop使其产生多样性
        up(512, 4, apply_drop=True),      # 4*4*512
        up(512, 4, apply_drop=True),      # 8*8*512
        up(512, 4),                        # 16*16*512
        up(256, 4),                        # 32*32*256
        up(128, 4),                        # 64*64*128
        up(64, 4),                        # 128*128*64
        # up(128, 4),                        # 256*256*128
        # up(64, 4),                        # 512*512*128
    ]
    
    x = inputs
    
    skips = []     
    
    for d in down_stack:
        x = d(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])  # 去掉最后一层翻转
    
    for u, skip in zip(up_stack, skips):
        x = u(x)
        x = tf.keras.layers.Concatenate()([x, skip])    # x为 512*512*128
        
    x = tf.keras.layers.Conv2DTranspose(3, 4, strides=2,  # x为 1024*1024*3
                                        padding='same', 
                                        activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# In[47]:


generator = Generator()


def Discriminator():
    inp = tf.keras.layers.Input(shape=(256, 256, 9))
    tar = tf.keras.layers.Input(shape=(256, 256, 3))
    
    x = tf.keras.layers.concatenate([inp, tar])   # 256, 256, 12
    
    x = down(64, 4, apply_bn=False)(x)               # 128, 128, 64
    x = down(128, 4)(x)                              # 64, 64, 128
    x = down(256, 4)(x)                              # 32, 32, 256
    # x = down(256, 4)(x)                              # 32, 32, 256
    # x = down(256, 4)(x)                              # 32, 32, 256
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1,
                               padding='same',
                               use_bias=False)(x)    # 32, 32, 512
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(1, 3, strides=1)(x)   # 30, 30, 1
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


#

discriminator = Discriminator()


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


LAMBDA = 10


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


count = 0


def generate_images(model, test_input, tar):
    global count
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [tar[0], prediction[0]]
    title = ['Real Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
#
    plt.savefig(r"D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\reslut\pix2pix原始\结果\对比\{}.png".format(count), bbox_inches="tight",
                pad_inches=0.0)

    plt.figure(figsize=(10, 10))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(tar[0]))
    plt.axis('off')
    plt.savefig(r"D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\reslut\pix2pix原始\结果\真实\{}.png".format(count),
                bbox_inches="tight", pad_inches=0.0)
    plt.figure(figsize=(10, 10))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(prediction[0]))
    plt.axis('off')
    plt.savefig(r"D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\reslut\pix2pix原始\结果\预测\{}.png".format(count),
                bbox_inches="tight", pad_inches=0.0)
    count+=1


EPOCHS = 6


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)  # 生成器输出（256,256,3）

        disc_real_output = discriminator([input_image, target], training=True)  # 判别器输出（30，30,1）
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    return gen_l1_loss, gen_gan_loss, gen_total_loss, disc_loss


# In[60]:


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs+1):
        if epoch % 3 == 0 and epoch != 0:
            for example_input, example_target in test_ds.take(len(image_T_I)):
                generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for n, (input_image, target) in train_ds.enumerate():
            # print('.', end='')
            # print('shape',np.shape(input_image))
            a, b, c, d = train_step(input_image, target, epoch)
            sum1 = sum1+a
            sum2 = sum2+b
            sum3 = sum3+c
            sum4 = sum4+d
        if epoch % 3 == 0 and epoch != 0:
            generator.save(r'D:\资料库\北京工商大学研究生学习资料\李文浩3DGAN\reslut\pix2pix原始\generator\gen%d.h5' % epoch)
            

fit(dataset, EPOCHS, dataset_T)


time_end = time.time()
print('time cost', time_end-time_start, 's')






