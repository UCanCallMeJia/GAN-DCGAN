'''
Implementation of GAN & DCGAN
with tensorflow/Keras.

Change the G/D models to 
implement GAN & DCGAN.

2019-11-20
@jiazx@buaa.edu.cn
'''
from keras.datasets import mnist
from keras.layers import Input,Dropout,Reshape,Dense,Conv2DTranspose
from keras.layers import Flatten,BatchNormalization,Activation,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam,SGD
from PIL import Image
import numpy as np  
import math

# Input shape:
IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# z noise size
LATENT_DIM = 100
OP = Adam(lr=0.0001, beta_1=0.5)


def build_generator():
    
    model = Sequential()
    model.add(Dense(128*7*7, activation='relu', input_shape=(LATENT_DIM,)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((7,7,128)))
    
    model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters=32, kernel_size=(3,3), strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters=CHANNELS, kernel_size=(3,3), strides=1, padding='same'))
    # DCGAN 的最后一层为tanh，限制输出在0-1之间
    model.add(Activation('tanh'))
    model.summary()

    return model

def build_generator_upsampling():
    
    model = Sequential()
    model.add(Dense(128*7*7, activation='relu', input_shape=(LATENT_DIM,)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((7,7,128)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    # model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    # DCGAN 的最后一层为tanh，限制输出在0-1之间
    model.add(Activation('tanh'))

    model.summary()

    return model


def build_discriminator():
    # 28*28*1-->14*14*64-->7*7*128-->4*4*256-->4*4*512
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='same', input_shape=IMG_SHAPE))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), strides=2, padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    return model

def build_adversarial_model(d, g):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def train(epochs, batch_size=128, save_interval=100):
 
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float)-127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=3)

    # 判别器
    d = build_discriminator()
    # 生成器
    g = build_generator_upsampling()
    # 冻结判别器后的 生成对抗网络
    a = build_adversarial_model(d, g)

    optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # 编译模型
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    a.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    

    for epoch in range(epochs):

        for i in range(int(x_train.shape[0]/batch_size)):
        
            ################  训练判别器  ################

            noises = np.random.uniform(-1, 1, size=(batch_size, LATENT_DIM))
            generated_imgs = g.predict(noises)
            g_imgs_labels = [0]*batch_size
            imgs = x_train[i*batch_size : (i+1)*batch_size]
            imgs_labels = [1]*batch_size
            x_d = np.concatenate((imgs, generated_imgs))
            y_d = np.concatenate((imgs_labels, g_imgs_labels))
            # 检查判别器网络参数是否可训练
            # d.summary()
            # 训练判别器
            d_loss = d.train_on_batch(x_d, y_d)
            # 训练后冻结参数
            d.trainable = False

            ################  训练生成器  ################
            
            noises = np.random.uniform(-1, 1, size=(batch_size, LATENT_DIM))
            noises_labels = [1]*batch_size
            # 检查判别器网络是否被冻结
            # a.summary()
            # 训练生成器
            a_loss = a.train_on_batch(noises, noises_labels)
            # 解冻判别器网络
            d.trainable = True

        # 保存生成的图片
        image = combine_images(generated_imgs)
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save('./DCGAN/results_upsampling/'+str(epoch)+'.png')

        # 打印损失
        print('epoch:{},  \td_loss:{},  \td_acc:{},  \ta_loss:{},  \ta_acc:{}'.
                format(epoch, d_loss[0], d_loss[1], a_loss[0], a_loss[1]))

if __name__ == "__main__":
    train(600)