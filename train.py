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


# Parâmetros
path = 'Classificador - Placas de Trânsito/dataset' # pasta com as imagens e classes
print(path)
label_file = 'Classificador - Placas de Trânsito/labels.csv' #arquivo com os nomes das classes
batch_size = 50 
steps_per_epoch = 1000 # Para melhores resultados, recomendo colocar 2000 
epochs = 10 # # Para melhores resultados, recomendo colocar 30
image_dim = (32, 32, 3)
test_ratio = 0.2 # 20% das imagens para teste
validation_ratio = 0.2 # 20% das imagens para validação

# Importação das imagens
count = 0
images = []
num_classes = []
class_list = os.listdir(path)
num_classes_len = len(class_list)

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


# Separação dos dados
x_train, x_test, y_train, y_test = train_test_split(images, num_classes, test_size=test_ratio, random_state=7)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

print('Treino:', x_train.shape, y_train.shape)
print('Teste: ', x_test.shape, y_test.shape)
print('Validação:', x_validation.shape, y_validation.shape)

# Arquivo .csv com as classes
data = pd.read_csv(label_file)
print(data.shape, type(data))

# Exibição de algumas imagens de todas as classes
num_samples = []
# cols = 5
# num_classes = num_classes_len
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

# Gráfigo de barras com o número de imagens de cada categoria
# plt.figure(figsize=(12, 5))
# plt.bar(range(0, num_classes_len), num_samples)
# plt.title('Distruição dos dados de treino')
# plt.xlabel('Número de classes')
# plt.ylabel('Número de imagens')

# plt.show()


# Pré processamento das imagens
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255  # normaliza as imagens entre 0 e 1 ao invés de 0 a 255
    return img


x_train = np.array(list(map(preprocessing, x_train)))
x_validation = np.array(list(map(preprocessing, x_validation)))
x_test = np.array(list(map(preprocessing, x_test)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # adiciona depth 1
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Geração de mais imagens (data augmentation)
data_gen = ImageDataGenerator(width_shift_range=0.1,  # 10%
                              height_shift_range=0.1,
                              zoom_range=0.2,
                              shear_range=0.1,
                              rotation_range=10)
data_gen.fit(x_train)
batches = data_gen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

y_train = to_categorical(y_train, num_classes_len)
y_validation = to_categorical(y_validation, num_classes_len)
y_test = to_categorical(y_test, num_classes_len)


print('Treino:', y_train.shape)
print('Validação:', y_validation.shape)
print('Teste:', y_test.shape)


# Criação da CNN
def conv_model():
    num_filters = 60
    filter_size = (5, 5)
    filter_size_2 = (3, 3)
    pool_size = (2, 2)
    num_nodes = 500

    model = Sequential()
    model.add(Conv2D(num_filters, filter_size, input_shape=(image_dim[0], image_dim[1], 1), activation='relu'))
    model.add(Conv2D(num_filters, filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(num_filters//2, filter_size_2, activation='relu'))
    model.add(Conv2D(num_filters//2, filter_size_2, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_nodes, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes_len, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Treinamento
model = conv_model()
print(model.summary())
history = model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=(x_validation, y_validation),
                            shuffle=7)

# Resultados
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Treinamento', 'Validação'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Treinamento', 'Validação'])
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Pontuação de teste:', score[0])
print('Acurácia de teste:', score[1])

# Salvamento do modelo
final_model = open('final_model.p', 'wb')
pickle.dump(model, final_model)
final_model.close()
cv2.waitKey(0)