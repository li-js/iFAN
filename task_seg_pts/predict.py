import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Deconvolution2D, Dropout, Flatten, Reshape
import time, cv2
import pickle

import matplotlib
matplotlib.use('Agg')
from pylab import plt

import numpy as np
from PIL import Image
import random

sys.path.append('./libs')
import decoders_def
import encoders_def
import densenet_fc

import glob
img_list=glob.glob('../demo/*.jpg')

batch_size = 3

seg_pretrain='model_weights/model_seg.h5'
pts_pretrain='model_weights/model_pts.h5'


w0,h0=128,128
ITER=3

nb_dense_block=5
encoder_seg, encoder_pts=densenet_fc.create_fc_dense_net(nb_classes=0, img_dim=(h0,w0,3), 
                                hidden1_input_dim=(h0,w0,9),hidden2_input_dim=(h0,w0,5),
                                nb_dense_block=nb_dense_block, growth_rate=12, nb_filter=16, nb_layers=3)

pred_pts  =decoders_def.pts_decoder(5*2)(Flatten()(encoder_pts.get_layer('merge_49').output))
pred_seg = decoders_def.seg_decoder(9, h0, w0)(encoder_seg.get_output_at(0))


model_pts = Model(input=encoder_pts.input, output=pred_pts)
model_seg = Model(input=encoder_seg.input, output=pred_seg)

model_pts.load_weights(pts_pretrain)
model_seg.load_weights(seg_pretrain)


def get_hidden_data_list(given_seg, given_pts):
    seg_encoded_2=encoders_def.encode_seg(given_seg, h0, w0)
    pts_encoded_2=encoders_def.encode_pts(given_pts, h0, w0, radius=5)
    return [seg_encoded_2, pts_encoded_2]


import pickle
Mean_Values=pickle.load(open("model_weights/mean_values.pickle"))
seg_pts_encoded=get_hidden_data_list(np.tile(Mean_Values['mean_seg'], (batch_size,1,1,1)), 
    np.tile(Mean_Values['mean_pts'],(batch_size,1)))


Imgs=np.zeros((batch_size,h0,w0,3),dtype=np.float32)
for kk in xrange(batch_size):
    img_add=img_list[kk]
    img = image.img_to_array(image.load_img(img_add, target_size=(h0, w0)))
    Imgs[kk]=img

Image_data = preprocess_input(Imgs)
hidden_list_list=[seg_pts_encoded]
pred_seg_list=[]
pred_pts_list=[]
for iter_k in xrange(ITER):
    pred_seg = model_seg.predict([Image_data]+hidden_list_list[iter_k])
    pred_pts = model_pts.predict([Image_data]+hidden_list_list[iter_k])
    pred_seg_list.append(pred_seg)
    pred_pts_list.append(pred_pts)  
    if iter_k<ITER-1:
        hidden_list_list.append(get_hidden_data_list(pred_seg.reshape((-1, h0, w0, 9)), pred_pts))


pred_pts=pred_pts_list[-1]
pred_seg=pred_seg_list[-1]

for kk in xrange(batch_size):
    im_add=img_list[kk]
    print im_add
    img_origin=np.array(Image.open(im_add))

    pred_seg_i=pred_seg[kk].reshape((h0,w0,9))
    pred_seg_i_origin=cv2.resize(pred_seg_i, (img_origin.shape[1], img_origin.shape[0]))
    pred_seg_im=pred_seg_i_origin.argmax(axis=2).astype('int8')

    pred_pts_i=pred_pts[kk].reshape((2,5))
    pts=(pred_pts_i+0.5)*np.array((img_origin.shape[1], img_origin.shape[0])).reshape((2,1))

    plt.clf()
    plt.figure(1)   
    plt.imshow(img_origin)
    plt.scatter(pts[0],pts[1], color='white', s=50)
    plt.savefig(img_list[kk].split('/')[-1]+'_pts.png')

    plt.clf()
    plt.figure(2)
    plt.imshow(pred_seg_im)
    plt.gca().axis('off')
    plt.savefig(img_list[kk].split('/')[-1]+'_seg.png')


#0 background
#1 facial skin
#2 eye brow
#3 eye
#4 nose
#5 upper lip
#6 within mouth
#7 lower lip
#8 hair

