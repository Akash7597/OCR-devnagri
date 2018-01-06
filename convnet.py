#!/usr/bin/env python
"""
modelapi.py
Author: Krishna Acharya
Date created: 21 Nov 2017
Python Version: 2.7
References:
Preprocessing ----> https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
Models----> https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import glob
np.random.seed(42)
current_dir = os.path.dirname(os.path.abspath(__file__))
learning_rate = 0.001

def get_network_arch():
	'''
		Uses the model providied at https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
	'''
	model = Sequential()        
	model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1),activation='relu', input_shape=(64,64,1)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer=Adam(lr=learning_rate), metrics=[categorical_accuracy])
	return model

def train():
	X, Y = load_data()
	X = X.astype('float32')
	X /= 255.0
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
	X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size = 0.5)
	augment_batch_size = 16        
	datagen = ImageDataGenerator(
	    rotation_range=10,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    shear_range=0.1,
	    zoom_range=0.1,
	    horizontal_flip=False,
	    fill_mode='nearest')
	        
	train_generator = datagen.flow(X_train, Y_train, batch_size=augment_batch_size) # here batch size is the number of augmentations
	validation_generator = datagen.flow(X_validate, Y_validate, batch_size=augment_batch_size)

	if os.path.isfile(current_dir + '/new_model.h5'):
	    model = load_model('new_model.h5')
	else:
	    model = get_network_arch()        

	model.fit_generator(
	            train_generator,
	            steps_per_epoch = 500,
	            epochs=20,
	            validation_data=validation_generator,
	            validation_steps = 40)
	            
	model.save('new_model.h5')
	score, acc = model.evaluate(X_test, Y_test, verbose=1)
	print 'Test score: ', score, 'Test accuraccy: ', acc

	print "##Exact match acc##"
	pred = model.predict(X_test)
	pred[pred>=0.5] = 1
	pred[pred<0.5] = 0
	print accuracy_score(Y_test, pred)

def multi_encode_128(filename):
	indices = np.array(map(int, filename.split('_')[3:])) - 2304  # 2304 corresponds to chandrabindu, every +16 goes to next column
	y_label = np.zeros(128)
	y_label[indices] = 1
	return y_label

def multi_decode_128(y_label):
	dec = np.where(y_label==1)[0] + 2304
	labels=[]
	for int_val in dec:
		labels.append(int_val)
	return labels
def preprocess(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	resized_im = cv2.resize(th3, (64,64))
	return resized_im

def load_data():
	X, Y = [],[]
	files  =[f for f in os.listdir(current_dir + '/train_images') if f.endswith('.png')]
	for file in files:
		y_label = multi_encode_128(file[:-4])
		Y.append(y_label)
		img = cv2.imread(current_dir+'/train_images/'+file, 0)
		X.append(preprocess(img).reshape(64,64,1)) # tensorflow backend needs this format
	X, Y = np.array(X), np.array(Y)
	return X, Y

def predict(model, filename):
	img = cv2.imread(filename, 0)
	img = preprocess(img).astype('float32')
	img /= 255.0
	img = img.reshape((1,64,64,1)) # since we're predicting only on 1 datapoint so (1,..)
	pred = model.predict(img)
	pred[pred>=0.5] = 1
	pred[pred<0.5] = 0
	return multi_decode_128(pred[0])

# def try_on_unknown():
# 	model = load_model('saved_model.h5')
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample1.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample2.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample3.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample4.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample5.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample6.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample7.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample8.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample9.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample10.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample11.png', 0)
# 	predict(model,img)    
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample12.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample13.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample14.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample15.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample16.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample17.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample18.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample19.png', 0)
# 	predict(model,img)
# 	img = cv2.imread(current_dir +'/test_images/' + 'sample20.png', 0)
# 	predict(model,img)
#train()
#try_on_unknown()