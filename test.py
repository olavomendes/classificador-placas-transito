import cv2
import pickle
import numpy as np

# Configurações da webcam
frame_w = 640
frame_h = 480
brightness = 180
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)
cap.set(10, brightness)

# Importação do modelo
pickle_in = open('Classificador - Placas de Trânsito/final_model.p', 'rb')
model = pickle.load(pickle_in)

# Pré processamento da webcam
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

def getClassName(class_number):
    if   class_number == 0: return 'Limite de velocidade 20 km/h'
    elif class_number == 1: return 'Limite de velocidade 30 km/h'
    elif class_number == 2: return 'Limite de velocidade 50 km/h'
    elif class_number == 3: return 'Limite de velocidade 60 km/h'
    elif class_number == 4: return 'Limite de velocidade 70 km/h'
    elif class_number == 5: return 'Limite de velocidade 80 km/h'
    elif class_number == 6: return 'Fim do Limite de velocidade 80 km/h'
    elif class_number == 7: return 'Limite de velocidade 100 km/h'
    elif class_number == 8: return 'Limite de velocidade 120 km/h'
    elif class_number == 9: return 'Não ultrapasse'
    elif class_number == 10: return 'Não ultrapasse for vechiles over 3.5 metric tons'
    elif class_number == 11: return 'Vire à direita na próxima intersecção'
    elif class_number == 12: return 'Via prioritária'
    elif class_number == 13: return 'Dê a preferência'
    elif class_number == 14: return 'Pare'
    elif class_number == 15: return 'Sem veículos'
    elif class_number == 16: return 'Proibido veículos acima de 3.5 toneladas'
    elif class_number == 17: return 'Não entre'
    elif class_number == 18: return 'Cuidado'
    elif class_number == 19: return 'Curva perigosa à esquerda'
    elif class_number == 20: return 'Curva perigosa à direita'
    elif class_number == 21: return 'Curva dupla'
    elif class_number == 22: return 'Estrada acidentada'
    elif class_number == 23: return 'Estrada escorregadia'
    elif class_number == 24: return 'Caminho estreito à direita'
    elif class_number == 25: return 'Obras'
    elif class_number == 26: return 'Sinal de trânsito'
    elif class_number == 27: return 'Pedestres'
    elif class_number == 28: return 'Passagem de crianças'
    elif class_number == 29: return 'Passagem de bicicletas'
    elif class_number == 30: return 'Cuidado com o gelo/neve'
    elif class_number == 31: return 'Passagem de animais selvagens'
    elif class_number == 32: return 'Fim de todos os limites de velocidade'
    elif class_number == 33: return 'Vire à direita'
    elif class_number == 34: return 'Vire à esquerda'
    elif class_number == 35: return 'Siga em frente'
    elif class_number == 36: return 'Siga reto ou vire à direita'
    elif class_number == 37: return 'Siga reto ou à esquerda'
    elif class_number == 38: return 'siga à direita'
    elif class_number == 39: return 'siga à esquerda'
    elif class_number == 40: return 'Rotatória'
    elif class_number == 41: return 'Fim do Não ultrapasse'
    elif class_number == 42: return 'Fim do Não ultrapasse veículos acima de 3.5 toneladas'


while True:

    success, original_img = cap.read()

    # processamento da imagem
    img = np.asarray(original_img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow('Imagem processada', img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(original_img, 'Classe: ', (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(original_img, 'Probabilidade: ', (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Previsão
    pred = model.predict(img)
    class_index = model.predict_classes(img)
    prob_value = np.amax(pred)

    if prob_value > threshold:
        cv2.putText(original_img, str(class_index) + ' '+ str(getClassName(class_index)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(original_img, str(round(prob_value*100,2) ) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Resultado', original_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()