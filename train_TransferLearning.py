#!/usr/bin/env python
# title           :train.py
# description     :to train the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python train.py --options
# python_version  :3.5.4
from keras.engine.saving import load_model

from Network import Generator, Discriminator
import Utils_model, Utils
from Utils_model import VGG_LOSS

from keras.models import Model
from keras.layers import Input, Conv3D
from tqdm import tqdm
import numpy as np
import argparse

np.random.seed(10)
# Better to use downscale factor as 4
downscale_factor = 2
# Remember to change image shape if you are having different size of images
# image_shape = (41, 41, 41, 1)
# dis_shape = (41, 41, 41, 1)
image_shape = (21, 21, 21, 1)
dis_shape = (21, 21, 21, 1)

# Combined network
# def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
#     discriminator.trainable = False
#     gan_input = Input(shape=shape)
#     x = generator(gan_input)
#     gan_output = discriminator(x)
#     gan = Model(inputs=gan_input, outputs=[x,gan_output])
#     gan.compile(loss=[vgg_loss, "binary_crossentropy"],
#                 loss_weights=[1., 1e-3],
#                 optimizer=optimizer)
#
#     return gan

# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, input_dir, tgt_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, saved_model):
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(input_dir, tgt_dir, '.npy',
                                                                            number_of_images, train_test_ratio)
    # x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(input_dir, '.jpg', number_of_images, train_test_ratio)
    # x_train_hr = np.expand_dims(x_train_hr, axis=3)
    # x_test_hr = np.expand_dims(x_test_hr, axis=3)
    # x_train_hr = np.reshape(x_train_hr,(x_train_hr[0], x_train_hr[1], x_train_hr[2], 1))
    # x_test_hr = np.reshape(x_test_hr, (x_test_hr[0], x_test_hr[1], x_test_hr[2], 1))

    loss = VGG_LOSS(image_shape)

    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0], image_shape[1], image_shape[2], image_shape[3])
    #
    # generator = Generator(shape).generator()
    # # model = squeeze(Activation('tanh')(model), 4)
    #
    # # discriminator = Discriminator(dis_shape).discriminator()

    generator = Generator(shape).generator()
    generator_old = load_model(saved_model, custom_objects={'vgg_loss': loss.vgg_loss})
    x_tmp = generator_old.layers[-2].output
    # generator_old.layers.pop()
    # generator_old.layers.pop()
    # for layers in generator_old.layers:
    #     layers.trainable = False
    # define new layrs
    # x1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")
    # x2 = Conv2D(filters=1, kernel_size=5, strides=1, padding="same")
    # x3 = Activation('tanh')
    # generator = Concatenate()[generator_old, x1,x2,x3]

    # gan_input = Input(shape=shape)
    # x_tmp = mid_out#generator_old(gan_input)
    # x_tmp = Conv3D(filters=64, kernel_size=9, strides=1, padding="same", name="TransConv2d_1")(x_tmp)
    x_tmp = Conv3D(filters=128, kernel_size=5, strides=1, padding="same", name="TransConv2d_2")(x_tmp)
    x_tmp = Conv3D(filters=32, kernel_size=3, strides=1, padding="same", name="TransConv2d_3")(x_tmp)
    x_tmp = Conv3D(filters=1, kernel_size=1, strides=1, padding="same", name="TransConv2d_4")(x_tmp)
    # gan_output = Activation('tanh')(x_tmp)
    generator = Model(inputs=generator_old.input, outputs=x_tmp)

    # fine tune the layers.
    for layers in generator.layers[:-4]:
        layers.trainable = False

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    # discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    # gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')
    loss_file.close()

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_loss = generator.train_on_batch(image_batch_lr, image_batch_hr)

        # print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt', 'a')
        loss_file.write('epoch%d : Resnet_loss = %s ; \n' % (e, gan_loss))
        loss_file.close()

        if e == 1 or e % 5 == 0:
            rand_nums = np.random.randint(0, x_test_hr.shape[0], size=batch_size)
            image_batch_hr = x_test_hr[rand_nums]
            image_batch_lr = x_test_lr[rand_nums]
            test_loss = generator.test_on_batch(image_batch_lr, image_batch_hr)
            print("test_loss :", test_loss)
            test_loss = str(test_loss)
            loss_file = open(model_save_dir + 'test_losses.txt', 'a')
            loss_file.write('epoch%d : test_loss = %s ; \n' % (e, test_loss))
            loss_file.close()
        if e % 50 == 0:
            generator.save(model_save_dir + 'Resnet_model%d.h5' % e)
            # discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/Simulation_Proj/',
                        help='Path for input images')

    parser.add_argument('-tgt', '--tgt_dir', action='store', dest='tgt_dir',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/Simulation_Proj/',
                        help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
                        help='Path for model')

    parser.add_argument('-oldm', '--model_dir', action='store', dest='model_dir',
                        default='./model/Resnet_Base-model30.h5',
                        help='Path for model')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=32,
                        help='Batch Size', type=int)

    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=1000,
                        help='number of iteratios for trainig', type=int)

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=3000,
                        help='Number of Images', type=int)

    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.90,
                        help='Ratio of train and test Images', type=float)

    values = parser.parse_args()

    train(values.epochs, values.batch_size, values.input_dir, values.tgt_dir, values.output_dir, values.model_save_dir,
          values.number_of_images, values.train_test_ratio, values.model_dir)


