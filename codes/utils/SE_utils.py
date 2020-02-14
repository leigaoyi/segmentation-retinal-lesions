# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:16:45 2020

@author: kasy
"""

import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose, \
    MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,\
    BatchNormalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

from keras.layers.merge import Concatenate

from utils.sequzze_excite import squeeze_excite_block
from se_utils import  _tensor_shape

#======================== UNet=================================
#==============================================================
def get_unet(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (unet)
    """
    axis = 3
    k = 3 # kernel size
    s = 2 # stride
    n_filters = 32 # number of filters

    inputs = Input((patch_height, patch_width, channels))
    conv1 = Conv2D(n_filters, (k,k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s,s))(conv1)

    conv2 = Conv2D(2*n_filters, (k,k), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s,s))(conv2)

    conv3 = Conv2D(4*n_filters, (k,k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k,k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)

    return unet


#===================================================
#====================Attention UNet=================

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
    # upsample_psi=my_repeat([upsample_psi])
    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    # print(K.is_keras_tensor(upsample_psi))

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UnetGatingSignal(input, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UnetConv2D(input, outdim, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def get_attention(patch_height, patch_width, channels, n_classes):

    n_filters = 32

    inputs = Input((patch_height, patch_width, channels))
    #conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
    #conv = LeakyReLU(alpha=0.3)(conv)

    conv1 = UnetConv2D(inputs, n_filters, is_batchnorm=True)  # 32 128
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 2*n_filters, is_batchnorm=True)  # 32 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 4*n_filters, is_batchnorm=True)  # 64 32
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 8*n_filters, is_batchnorm=True)  # 64 16
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 16*n_filters, is_batchnorm=True)  # 128 8

    gating = UnetGatingSignal(center, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 8*n_filters)
    up1 = concatenate([Conv2DTranspose(8*n_filters, (3, 3), strides=(2, 2), padding='same', activation="relu")(center), attn_1],
                      axis=3)
    conv5 = UnetConv2D(up1, 8*n_filters, is_batchnorm=True)

    gating = UnetGatingSignal(conv5, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 4*n_filters)
    up2 = concatenate([Conv2DTranspose(4*n_filters, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv5), attn_2],
                      axis=3)
    conv6 = UnetConv2D(up2, 4*n_filters, is_batchnorm=True)

    gating = UnetGatingSignal(conv6, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 2*n_filters)
    up3 = concatenate([Conv2DTranspose(2*n_filters, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv6), attn_3],
                      axis=3)
    conv7 = UnetConv2D(up3, 2*n_filters, is_batchnorm=True)

    up4 = concatenate([Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv7), conv1],
                      axis=3)
    conv8 = UnetConv2D(up4, n_filters, is_batchnorm=True)

    conv9 = Conv2D(n_classes , (1, 1), activation='softmax', padding='same')(conv8)
    #conv8 = core.Reshape((patch_height * patch_width, (n_classes + 1)))(conv8)
    ############
    #act = Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
    #self.model = model
    return model


def get_att_unet(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (unet)
    """
    axis = 3
    k = 3 # kernel size
    s = 2 # stride
    n_filters = 32 # number of filters

    inputs = Input((patch_height, patch_width, channels))
    conv1 = Conv2D(n_filters, (k,k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s,s))(conv1)

    conv2 = Conv2D(2*n_filters, (k,k), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s,s))(conv2)

    conv3 = Conv2D(4*n_filters, (k,k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    
    gating = conv5
    attn_1 = AttnGatingBlock(conv4, gating, 16*n_filters)

    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv5), attn_1])
    conv6 = Conv2D(8 * n_filters, (k,k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    
    gating = conv6
    attn_2 = AttnGatingBlock(conv3, gating, 8*n_filters)

    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), attn_2])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    
    gating = conv7
    attn_3 = AttnGatingBlock(conv2, gating, 4*n_filters)

    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), attn_3])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)

    return unet

#=================================================================
#=============================SE-Res-UNet=========================



is_keras_tensor = K.is_keras_tensor
TF = False


#def res_block(x, nb_filters, strides):
#    res_path = BatchNormalization()(x)
#    res_path = Activation(activation='relu')(res_path)
#    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
#    res_path = BatchNormalization()(res_path)
#    res_path = Activation(activation='relu')(res_path)
#    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)
#
#    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
#    shortcut = BatchNormalization()(shortcut)
#
#    res_path = add([shortcut, res_path])
#    return res_path



def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], (2, 2))
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], (2, 2))
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], (1, 1))

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], (1, 1))

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], (1, 1))

    return main_path


def res_block(input_tensor, filters, strides=(1, 1), k=1):
    """ Adds a pre-activation resnet block without bottleneck layers
    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a Keras tensor
    """
    init = input_tensor
    filters = filters[0]
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1,1) or _tensor_shape(init)[channel_axis] != filters * k:
        #print('strides 2*************')
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    
    x = squeeze_excite_block(x)
    #print('x shape', x.shape)
    m = add([x, init])
    return m


def _resnet_bottleneck_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block with bottleneck layers
    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m

def build_res_unet(patch_height, patch_width, channels, n_classes):
    inputs = Input((patch_height, patch_width, channels))

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [256, 256], (2, 2))

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='softmax', padding='SAME')(path)

    return Model(input=inputs, output=path)

