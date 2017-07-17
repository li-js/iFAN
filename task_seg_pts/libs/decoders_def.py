from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, Dropout,Flatten, Reshape, Activation, BatchNormalization, merge
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU

def seg_decoder(nb_classes, h0,w0, weight_decay=1E-4):
	def f(input):
		x = Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(weight_decay),bias=False)(input)
		x = Reshape((w0*h0, nb_classes))(x)
		return Activation('softmax', name='pred_seg')(x)
	return f

def ptsmap_decoder(nb_classes, h0,w0, weight_decay=1E-4):
	def f(input):
		x = Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(weight_decay),bias=False)(input)
		x = Reshape((w0*h0, nb_classes))(x)
		return Activation('softmax', name='pred_ptsmap')(x)
	return f

def pose_decoder(nb_classes, weight_decay=1E-4):
	def f(input):
		x = Dropout(0.3)(input)
		x = Dense(nb_classes, W_regularizer=l2(weight_decay))(x)
		return Activation('softmax',name='pred_pose')(x)
	return f

def emo_decoder(nb_classes, weight_decay=1E-4):
	def f(input):
		x = Dropout(0.3)(input)
		x = Dense(nb_classes, W_regularizer=l2(weight_decay))(x)
		return Activation('softmax',name='pred_emo')(x)
	return f


def pts_decoder(nb_classes, weight_decay=1E-4):
	def f(input):
		x = Dropout(0.3)(input)
		x = Dense(nb_classes, W_regularizer=l2(weight_decay),name='pred_pts')(x)
		return x
	return f