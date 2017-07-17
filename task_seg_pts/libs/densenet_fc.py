from keras.models import Model
from keras.layers.core import Activation, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from layers import SubPixelUpscaling

def conv_block(ip1, ip2, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional bottleneck block and dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x1 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip1)
    x2 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip2)
    ac=Activation('relu')
    x1 = ac(x1)
    x2 = ac(x2)

    if bottleneck:
        inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        conv_tmp=Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay))

        x1 = conv_tmp(x1)
        x2 = conv_tmp(x2)

        if dropout_rate:
            do=Dropout(dropout_rate)
            x1 = do(x1)
            x2 = do(x2)

        x1 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x1)
        x2 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x2)
        ac=Activation('relu')                           
        x1 = ac(x1)
        x2 = ac(x2)

    conv_tmp=Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))
    x1 = conv_tmp(x1)
    x2 = conv_tmp(x2)
    if dropout_rate:
        do=Dropout(dropout_rate)
        x1 = do(x1)
        x2 = do(x2)

    return x1,x2


def transition_down_block(ip1, ip2, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x1 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip1)
    x2 = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip2)
    ac=Activation('relu')                   
    x1 = ac(x1)
    x2 = ac(x2)
    conv_tmp=Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))
    x1 = conv_tmp(x1)
    x2 = conv_tmp(x2)

    if dropout_rate:
        do=Dropout(dropout_rate)
        x1 = do(x1)
        x2 = do(x2)
    ap=AveragePooling2D((2, 2), strides=(2, 2))
    x1 = ap(x1)
    x2 = ap(x2)

    return x1, x2


def transition_up_block(ip1, ip2, nb_filters, type='subpixel', output_shape=None, weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)

    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'subpixel' or 'deconv'. Determines type of upsampling performed
        output_shape: required if type = 'deconv'. Output shape of tensor
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    if type == 'subpixel':
        conv_tmp=Convolution2D(nb_filters, 3, 3, activation="relu", border_mode='same', W_regularizer=l2(weight_decay),
                        bias=False)
        x1 = conv_tmp(ip1)
        x2 = conv_tmp(ip2)
        sp = SubPixelUpscaling(r=2, channels=int(nb_filters // 4))
        x1 = sp(x1)
        x2 = sp(x2)
        conv_tmp=Convolution2D(nb_filters, 3, 3, activation="relu", border_mode='same', W_regularizer=l2(weight_decay),
                          bias=False)
        x1 = conv_tmp(x1)
        x2 = conv_tmp(x2)

    else:
        assert(0), "Not implemented"

    return x1,x2


def dense_block(x1, x2, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list1 = [x1]
    feature_list2 = [x2]

    for i in range(nb_layers):
        x1,x2 = conv_block(x1,x2, growth_rate, bottleneck, dropout_rate, weight_decay)
        feature_list1.append(x1)
        feature_list2.append(x2)
        x1 = merge(feature_list1, mode='concat', concat_axis=concat_axis)
        x2 = merge(feature_list2, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x1,x2, nb_filter


def create_fc_dense_net(nb_classes, img_dim, hidden1_input_dim, hidden2_input_dim, nb_dense_block=5, growth_rate=12, nb_filter=16, nb_layers=4, upsampling_conv=128,
                        bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4, upscaling_type='subpixel',
                        verbose=True):
    ''' Build the create_dense_net model

    Args:
        nb_classes: Number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Setting -1 indicates initial number of filters is 2 * growth_rate
        nb_layers: number of layers in each dense block. Can be an -1, a positive integer or a list

                   If -1, it computes the nb_layer from depth

                   If positive integer, a set number of layers per dense block

                   If list, nb_layer is used as provided.
                   Note that list size must be (nb_dense_block + 1)

        upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        upscaling_type: method of upscaling. Can be 'subpixel' or 'deconv'
        verbose: print the model type

    Returns: keras tensor with nb_layers of conv_block appended

    '''
    if K.backend() == 'tensorflow' and upscaling_type == 'deconv':
        assert len(img_dim) == 4, "If using tensorflow backend with deconvolution type upscaling, \n" \
                                  "then batch size must also be provided in img_dim as it is required for computing \n" \
                                  "output shape of the deconvolution layer"

        batch_size = img_dim[0]
        img_dim = img_dim[1:]

    else:
        batch_size = None

    img_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    if concat_axis == 1: # th dim ordering
        _, rows, cols = img_dim
    else:
        rows, cols, _ = img_dim

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert upsampling_conv > 12 and upsampling_conv % 4 == 0, "upsampling_conv number of channels must " \
                                                             "be a positive number divisible by 4 and greater " \
                                                              "than 12"

    assert upscaling_type.lower() in ['subpixel', 'deconv'], "upscaling_type must be either 'subpixel' or " \
                                                             "'deconv'"

    # layers in each dense block
    if type(nb_layers) is list or type(nb_layers) is tuple:
        nb_layers = list(nb_layers) # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), "If list, nb_layer is used as provided. " \
                                                        "Note that list size must be (nb_dense_block + 1)"

        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]

    else:
        final_nb_layer = nb_layers
        nb_layers = [nb_layers] * nb_dense_block

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    hidden1_input=Input(shape=hidden1_input_dim)
    hidden1_list=[]
    hidden1_list.append(hidden1_input)
    x = Convolution2D(24, 5, 5, init="he_uniform", subsample=(2,2),
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(hidden1_input)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden1_list.append(None)
    hidden1_list.append(x)

    x = Convolution2D(48, 3, 3, init="he_uniform", 
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden1_list.append(x)

    x = Convolution2D(64, 3, 3, init="he_uniform", 
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden1_list.append(x)
    hidden1_list.append(None)


    hidden2_input=Input(shape=hidden2_input_dim)
    hidden2_list=[]
    hidden2_list.append(hidden2_input)
    x = Convolution2D(24, 5, 5, init="he_uniform", subsample=(2,2),
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(hidden2_input)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden2_list.append(None)
    hidden2_list.append(x)

    x = Convolution2D(48, 3, 3, init="he_uniform", 
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden2_list.append(x)

    x = Convolution2D(64, 3, 3, init="he_uniform", 
        border_mode="same", bias=False, W_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(mode=0, axis=-1, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    hidden2_list.append(x)
    hidden2_list.append(None)

    # Initial convolution
    model_input=merge([img_input, hidden1_list[0], hidden2_list[0]],mode='concat')
    conv_init=Convolution2D(48, 7, 7, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))

    x_t1 = conv_init(model_input)
    x_t2 = conv_init(model_input)

    skip_connection_t1 = x_t1
    skip_connection_t2 = x_t2

    skip_list_t1 = []
    skip_list_t2 = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x_t1, x_t2, nb_filter = dense_block(x_t1, x_t2, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Skip connection
        x_t1 = merge([x_t1, skip_connection_t1], mode='concat', concat_axis=concat_axis)
        skip_list_t1.append(x_t1)
        x_t2 = merge([x_t2, skip_connection_t2], mode='concat', concat_axis=concat_axis)
        skip_list_t2.append(x_t2)
        if hidden1_list[block_idx] is not None: 
            x_t1 = merge([x_t1, hidden1_list[block_idx], hidden2_list[block_idx]], mode='concat', concat_axis=concat_axis)
            x_t2 = merge([x_t2, hidden1_list[block_idx], hidden2_list[block_idx]], mode='concat', concat_axis=concat_axis)

        # add transition_block
        x_t1, x_t2 = transition_down_block(x_t1, x_t2, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

        # Preserve transition for next skip connection after dense
        skip_connection_t1 = x_t1
        skip_connection_t2 = x_t2

    # The last dense_block does not have a transition_down_block
    x_t1, x_t2, nb_filter = dense_block(x_t1, x_t2, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        if hidden1_list[-(block_idx+1)] is not None: 
            x_t1 = merge([x_t1, hidden1_list[-(block_idx+1)],hidden2_list[-(block_idx+1)]], mode='concat', concat_axis=concat_axis)
            x_t2 = merge([x_t2, hidden1_list[-(block_idx+1)],hidden2_list[-(block_idx+1)]], mode='concat', concat_axis=concat_axis)
        x_t1, x_t2 = transition_up_block(x_t1, x_t2, nb_filters=upsampling_conv, type=upscaling_type)

        x_t1 = merge([x_t1, skip_list_t1.pop()], mode='concat', concat_axis=concat_axis)
        x_t2 = merge([x_t2, skip_list_t2.pop()], mode='concat', concat_axis=concat_axis)

        x_t1, x_t2, nb_filter = dense_block(x_t1,x_t2, nb_layers[-block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)

    x_t1 = merge([x_t1, hidden1_input,hidden2_input], mode='concat', concat_axis=concat_axis)
    x_t2 = merge([x_t2, hidden1_input,hidden2_input], mode='concat', concat_axis=concat_axis)

    if nb_classes>0:
        conv_tmp=Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(weight_decay),bias=False)
        x_t1 = conv_tmp(x_t1)
        x_t2 = conv_tmp(x_t2)

        if K.image_dim_ordering() == 'th':
            channel, row, col = img_dim
        else:
            row, col, channel = img_dim

        x_t1 = Reshape((row * col, nb_classes))(x_t1)
        x_t2 = Reshape((row * col, nb_classes))(x_t2)
        x_t1 = Activation('softmax')(x_t1)
        x_t2 = Activation('softmax')(x_t2)

    densenet_t1 = Model(input=[img_input, hidden1_input, hidden2_input], output=x_t1, name="create_dense_net_t1")
    densenet_t2 = Model(input=[img_input, hidden1_input, hidden2_input], output=x_t2, name="create_dense_net_t2")

    # Compute depth
    nb_conv_layers = len([layer.name for layer in densenet_t1.layers
                          if layer.__class__.__name__ == 'Convolution2D'])

    depth = nb_conv_layers -  nb_dense_block # For 1 extra convolution layers per transition up

    if verbose: print('Total number of convolutions', depth)

    if verbose:
        if bottleneck and not reduction:
            print("Bottleneck DenseNet-B-%d-%d created." % (depth, growth_rate))
        elif not bottleneck and reduction > 0.0:
            print("DenseNet-C-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        elif bottleneck and reduction > 0.0:
            print("Bottleneck DenseNet-BC-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        else:
            print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet_t1, densenet_t2


if __name__ == '__main__':
    from keras.utils.visualize_util import plot

    model = create_fc_dense_net(nb_classes=10, img_dim=(3, 224, 224), nb_dense_block=5, growth_rate=12,
                                nb_filter=16, nb_layers=4)
    model.summary()
    # plot(model, to_file='FC-DenseNet-56.png', show_shapes=False, show_layer_names=False)

    # print('\n\n\n\n\n\n\n\n\n')

    # model = create_fc_dense_net((3, 224, 224), bottleneck=False, reduction=0.5)
    # plot(model, to_file='FC-DenseNet-56.png', show_shapes=False, show_layer_names=False)

    # nb_layers = [4, 5, 7, 10, 12, 15]
    # model = create_fc_dense_net((3, 224, 224), nb_layers=nb_layers)
    # plot(model, to_file='FC-DenseNet-103.png', show_shapes=False, show_layer_names=False)

    # model = create_fc_dense_net((3, 224, 224), bottleneck=True, reduction=0.5)
    # model.summary()
    # plot(model, to_file='FC-DenseNet-56-BC.png', show_layer_names=False, show_shapes=False)