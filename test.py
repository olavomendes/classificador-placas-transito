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
    if   class_number == 0: return 'Speed Limit 20 km/h'
    elif class_number == 1: return 'Speed Limit 30 km/h'
    elif class_number == 2: return 'Speed Limit 50 km/h'
    elif class_number == 3: return 'Speed Limit 60 km/h'
    elif class_number == 4: return 'Speed Limit 70 km/h'
    elif class_number == 5: return 'Speed Limit 80 km/h'
    elif class_number == 6: return 'End of Speed Limit 80 km/h'
    elif class_number == 7: return 'Speed Limit 100 km/h'
    elif class_number == 8: return 'Speed Limit 120 km/h'
    elif class_number == 9: return 'No passing'
    elif class_number == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif class_number == 11: return 'Right-of-way at the next intersection'
    elif class_number == 12: return 'Priority road'
    elif class_number == 13: return 'Yield'
    elif class_number == 14: return 'Stop'
    elif class_number == 15: return 'No vechiles'
    elif class_number == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif class_number == 17: return 'No entry'
    elif class_number == 18: return 'General caution'
    elif class_number == 19: return 'Dangerous curve to the left'
    elif class_number == 20: return 'Dangerous curve to the right'
    elif class_number == 21: return 'Double curve'
    elif class_number == 22: return 'Bumpy road'
    elif class_number == 23: return 'Slippery road'
    elif class_number == 24: return 'Road narrows on the right'
    elif class_number == 25: return 'Road work'
    elif class_number == 26: return 'Traffic signals'
    elif class_number == 27: return 'Pedestrians'
    elif class_number == 28: return 'Children crossing'
    elif class_number == 29: return 'Bicycles crossing'
    elif class_number == 30: return 'Beware of ice/snow'
    elif class_number == 31: return 'Wild animals crossing'
    elif class_number == 32: return 'End of all speed and passing limits'
    elif class_number == 33: return 'Turn right ahead'
    elif class_number == 34: return 'Turn left ahead'
    elif class_number == 35: return 'Ahead only'
    elif class_number == 36: return 'Go straight or right'
    elif class_number == 37: return 'Go straight or left'
    elif class_number == 38: return 'Keep right'
    elif class_number == 39: return 'Keep left'
    elif class_number == 40: return 'Roundabout mandatory'
    elif class_number == 41: return 'End of no passing'
    elif class_number == 42: return 'End of no passing by vechiles over 3.5 metric tons'


while True:

    success, original_img = cap.read()

    # processamento da imagem
    img = np.asarray(original_img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow('Imagem processada', img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(original_img, 'Classe: ', (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(original_img, 'Probabilidade: ', (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Previsão
    pred = model.predict(img)
    class_index = model.predict_classes(img)
    prob_value = np.amax(pred)

    if prob_value > threshold:
        cv2.putText(original_img, str(class_index) + ' '+ str(getClassName(class_index)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(original_img, str(round(prob_value * 100, 2)) + '%'+  (180, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Resultado', original_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()