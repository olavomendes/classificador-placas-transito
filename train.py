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
label_file = 'Classificador - Placas de Trânsito/labels.csv' # arquivo com os nomes das classes
batch_size = 50 
steps_per_epoch = 2000
epochs = 30
image_dim = (32, 32, 3)
test_ratio = 0.2 # 20% das imagens para teste
validation_ratio = 0.2 # 20% das imagens para validação

#### Importação das imagens
count = 0
images = []
num_classes = []
class_list = os.listdir(path)
print('Total de classes: ', len(class_list))
print('Importando classes...')
for i in range(0, len(class_list)):
    image_list = os.listdir(path + '/' + str(count))
    for x in image_list:
        current_image = cv2.imread(path + '/' + str(count) + '/' + x)
        images.append(current_image)
        num_classes.append(count)
    print(count, end=' ')
    count += 1
print(' ')

images = np.array(images)
num_classes = np.array(num_classes)

#### Separação dos dados
x_train, x_test, y_train, y_test = train_test_split(images, num_classes, test_size=test_ratio, random_state=7)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

print('Treino:', x_train.shape, y_train.shape)
print('Validação:', x_validation.shape, y_validation.shape)

#### Arquivo .csv com as classes
data = pd.read_csv(label_file)
print(data.shape, type(data))

#### Exibição de algumas imagens de todas as classes
num_samples = []
# cols = 5
# num_classes = len(class_list)
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
# fig.tight_layout()

# for i in range(cols):
#     for j, row in data.iterrows():
#         x_selected = x_train[y_train == j]
#         axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap('gray'))
#         axs[j][i].axis('off')
#         if i == 2:
#             axs[j][i].set_title(str(j) + '-' + row['Name'])
#             num_samples.append(len(x_selected))

#### Gráfigo de barras com o número de imagens de cada categoria
plt.figure(figsize=(12, 5))
plt.bar(range(0, num_classes), num_samples)
plt.title('Distruição dos dados de treino')
plt.xlabel('Número de classes')
plt.ylabel('Número de imagens')

plt.show()