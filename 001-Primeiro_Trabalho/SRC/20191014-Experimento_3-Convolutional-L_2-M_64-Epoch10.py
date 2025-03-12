"""
COPPE/UFRJ (13/SET/2019)
Inteligencia Computacional II
 
Trabalho II - Trabalho Final
 
Nome: Luiz Marcio Faria de Aquino Viana
CPF: 024.723.347-10
RG: 08855128-8 IFP-RJ
 
003-Experimento_3-Convolutional-L_2-M_64-Epoch10
"""
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt 

import keras
import keras.callbacks

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.utils import to_categorical

from sklearn import datasets, svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from scipy.sparse import coo_matrix

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.acc = []
		self.list_X = []
		self.curr_X = 0
		self.list_Epoch = []
		self.list_Epoch.append(0.0)

	def on_batch_end(self, batch, logs={}):
		self.curr_X = self.curr_X + 1
		self.losses.append(logs.get('loss'))
		self.acc.append(logs.get('acc'))
		self.list_X.append(self.curr_X)

	def on_epoch_end(self, epoch, logs={}):
		self.list_Epoch.append(self.curr_X)

def calc_test(arr1, arr2):
	n = 0
	n_win = 0
	for v1 in arr1:
		v2 = arr2[n]
		n = n + 1
		if v1 == v2:
			n_win = n_win + 1
	result = (n_win / n * 100.0)
	return result

def plot_learning_curve(title, train_scores, train_sizes, train_epoch):
	plt.figure()
	plt.title(title)
	plt.xlabel("Training epoch")
	plt.ylabel("Score")
	plt.grid()
	plt.plot(train_sizes, train_scores)
	plt.legend(loc="best")
	n = 0
	for val_x in train_epoch:
		val_str = "Epoch " + str(n)
		plt.text(val_x, 0, val_str)
		n = n + 1
	return plt

startTime = datetime.datetime.now()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
 
n_train = len(X_train)
n_test = len(X_test)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Data Preparation - Elapsed Time")
print(elapsedTime)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
train_hist = LossHistory()

startTime = datetime.datetime.now()

hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1, callbacks=[train_hist])

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Train - Elapsed Time")
print(elapsedTime)

startTime = datetime.datetime.now()

accuracy = model.evaluate(X_test, y_test)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Classify and Test - Elapsed Time")
print(elapsedTime)

#for s_out in out:
#  print(s_out)

print("Result")
print(rst)

plot_learning_curve( 
	"L_2-M_64-Epoch_10-Losses", 
	train_hist.losses, 
	train_hist.list_X, 
	train_hist.list_Epoch)

plot_learning_curve( 
	"L_2-M_64-Epoch_10-Acc", 
	train_hist.acc, 
	train_hist.list_X,
	train_hist.list_Epoch)

plt.show()

