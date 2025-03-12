"""
COPPE/UFRJ (19/OUT/2019)
Inteligencia Computacional II
 
Trabalho II - Trabalho Final
 
Nome: Luiz Marcio Faria de Aquino Viana
CPF: 024.723.347-10
RG: 08855128-8 IFP-RJ
 
20191014-Analise_Base_NMINST_com_WISARD
"""

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

def convert_to_list(arr):
	lst = []
	for arr_row in arr:
		lst_row = []
		for arr_col in arr_row:
			lst_row.append(arr_col)
		lst.append(lst_row)
	return lst

def convert_to_list2(arr):
	lst = []
	for arr_col in arr:
		lst.append(arr_col)
	return lst

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

def show_mental_image(s):
	img = np.array(s, dtype='float')
	pixels = img.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.show()

import datetime
import wisardpkg as wp
import tensorflow as tf
import numpy as np
import math as m
import matplotlib.pyplot as plt 

from array import *
from matplotlib import pyplot as plt
from synthesizer import Player, Synthesizer, Waveform

startTime = datetime.datetime.now()

(X, y), (X_t, y_t) = tf.keras.datasets.mnist.load_data()

X = X.reshape(60000,784)
X_t = X_t.reshape(10000,784)

X = X / 128
X_t = X_t / 128

X = X.astype('int')
y = y.astype('str')

X_t = X_t.astype('int')
y_t = y_t.astype('str')

X1 = []
y1 = []

X_t1 = []
y_t1 = []

X1 = convert_to_list(X)
y1 = convert_to_list2(y)

X_t1 = convert_to_list(X_t)	
y_t1 = convert_to_list2(y_t)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Data Preparation - Elapsed Time")
print(elapsedTime)

#show_mental_image(X[0])
#show_mental_image(X[1])
#show_mental_image(X[2])
#show_mental_image(X[3])
#show_mental_image(X[4])
#show_mental_image(X[5])

wsd = wp.Wisard(3, ignoreZeros=False, verbose=False)

startTime = datetime.datetime.now()

wsd.train(X1, y1)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Train - Elapsed Time")
print(elapsedTime)

startTime = datetime.datetime.now()

out = wsd.classify(X_t1)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Classify - Elapsed Time")
print(elapsedTime)

#for s_out in out:
#  print(s_out)

startTime = datetime.datetime.now()

rst = calc_test(out, y_t1)

endTime = datetime.datetime.now()

elapsedTime = endTime - startTime
print("Test - Elapsed Time")
print(elapsedTime)

print("Result")
print(rst)

mentalImages = wsd.getMentalImages() 
s0 = mentalImages["0"]
s1 = mentalImages["1"]
s2 = mentalImages["2"]
s3 = mentalImages["3"]
s4 = mentalImages["4"]
s5 = mentalImages["5"]
s6 = mentalImages["6"]
s7 = mentalImages["7"]
s8 = mentalImages["8"]
s9 = mentalImages["9"]

#show_mental_image(s0)
#show_mental_image(s1)
#show_mental_image(s2)
#show_mental_image(s3)
#show_mental_image(s4)
#show_mental_image(s5)
#show_mental_image(s6)
#show_mental_image(s7)
#show_mental_image(s8)
#show_mental_image(s9)

