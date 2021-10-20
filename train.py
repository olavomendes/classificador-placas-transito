import random
import os
import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split


#### Parâmetros
path = 'Classificador - Placas de Trânsito/dataset' # pasta com as imagens e classes
print(path)
label_file = 'labels.csv' # arquivo com os nomes das classes
batch_size = 50 
steps_per_epoch = 2000
epochs = 30
image_dim = (32, 32, 3)
test_ratio = 0.2 # 20% das imagens para teste
validation_ratio = 0.2 # 20% das imagens para validação

#### Importação das imagens
count = 0
images = []
classes_numbers = []
class_list = os.listdir(path)
print('Total de classes: ', len(class_list))
print('Importando classes...')
for i in range(0, len(class_list)):
    image_list = os.listdir(path + '/' + str(count))
    for x in image_list:
        current_image = cv2.imread(path + '/' + str(count) + '/' + x)
        images.append(current_image)
        classes_numbers.append(count)
    print(count, end=' ')
    count += 1
print(' ')