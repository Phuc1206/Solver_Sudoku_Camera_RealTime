import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import realtimesudoku

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (1066, 600))
cap.set(3, 1280)    
cap.set(4, 720)
input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.load_weights("digitRecognition.h5")   

old_sudoku = None
while(True):
    ret, frame = cap.read() 
    if ret == True:
        sudoku_frame = realtimesudoku.recognize_and_solve_sudoku(frame, model, old_sudoku) 
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
# output.release()
cv2.destroyAllWindows()