#Import Library's
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import ctypes
import random
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D



# Training the Ai
train_input = input('Train?').lower()
if 'y' in train_input:
    Ai = pd.read_csv('PictionaryAi.csv')
    X_train, y_train = Ai.drop(['Label'], axis=1), Ai['Label']
    X_train = X_train.values.reshape(-1, 50, 50, 1)
    y_train = to_categorical(y_train, num_classes=Ai['Label'].nunique())
    Pic_Ai = Sequential()
    Pic_Ai.add(Conv2D(2500, (3,3), activation='relu', input_shape=(X_train.shape[1:4])))
    Pic_Ai.add(Flatten())
    Pic_Ai.add(Dense(Ai['Label'].nunique(), activation='softmax'))

    Pic_Ai.compile(optimizer='rmsprop', 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
    Pic_Ai.fit(X_train.reshape(-1,50,50,1), y_train.reshape(-1, Ai['Label'].nunique()),
               epochs=5, verbose=1);
# Make Random Word generator
words = ('fork', 'car', 'apple', 'book', 'tv', 'door', 'tree', 'mug', 'lamp', 'phone', 'outlet', 'soccer ball', 'clock', 'cartoon heart')
random_word = (random.choice(words))
messageBox = ctypes.windll.user32.MessageBoxW
print(random_word)
first_letter = random_word[0].lower()
Vowel = False
if first_letter == 'a' or first_letter == 'e':
    Vowel = True
    returnValue = messageBox(None, 'Please Draw an {}'.format(random_word), 'Sketch Draw', 0x70 | 0x1)
    if returnValue == 1:
        pass
    elif returnValue == 2:
        while True:
            cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
elif first_letter == 'i' or first_letter == 'o' or first_letter == 'u':
    Vowel = True
    returnValue = messageBox(None, 'Please Draw an {}'.format(random_word), 'Sketch Draw', 0x70 | 0x1)
    if returnValue == 1:
        pass
    elif returnValue == 2:
        while True:
            cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
elif not Vowel:
    returnValue_novowel = messageBox(None, 'Please Draw a {}'.format(random_word), 'Sketch Draw', 0x70 | 0x1)
    print(returnValue_novowel)
    if returnValue_novowel == 1:
        pass
    elif returnValue_novowel == 2:
        while True:
            cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
    # Give User 4 Options
    # User Can pick an option by clicking on it
brush = False
x1, y1 = None, None


def draw(event, x, y, four, five):
    global x1, y1, brush

    if event == cv2.EVENT_LBUTTONDOWN:
        brush = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if brush:
            cv2.line(img, (x1, y1), (x, y), (255, 255, 255), thickness=5)
            x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        brush = False
        cv2.line(img, (x1, y1), (x, y), (255, 255, 255), thickness=5)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('Pictionary')
cv2.setMouseCallback('Pictionary', draw)

while 1:
    cv2.imshow('Pictionary', img)
    if cv2.waitKey(1) & 0xFF == 27:
        image_load = False
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(33) == ord('a'):
        image_load = True
        break
cv2.destroyAllWindows()
# Get Data and print image
if image_load:
    cv2.imwrite('{}.png'.format(random_word), img)
    img_resize = Image.open('{}.png'.format(random_word))
    print(img_resize.size)
    img_resize = img_resize.resize((50, 50), Image.ANTIALIAS)
    plt.imshow(img_resize, cmap='Greys')
    plt.axis('off')
    plt.show()
    img_resize.save('{}.png'.format(random_word))
    df = np.array(Image.open('{}.png'.format(random_word))) / 255
    df = df[:, :, 0]
    df = df.reshape(50, 50, 1)
    cv2.imwrite('{}.png'.format(random_word), df)
img = Image.open('{}.png'.format(random_word))
img_df = np.array(img)
columns = []
i = 1
for num in range(2500):
    columns.append('Pixel_{}'.format(i))
    i += 1
X_test = pd.DataFrame(columns=columns, data=img_df.reshape(1,2500))
prediction = Pic_Ai.predict(img_df.reshape(1,50,50,1))
prediction = pd.DataFrame(data=prediction, columns=['Book', 'Lamp', 'Fork', 'Phone', 'Tv', 'Tree', 
                                            'Apple', 'Soccer Ball', 'Car', 'Mug', 'Outlet', 'Clock', 'Door', 'Heart'])
output = max(prediction.values.reshape(-1,1))
where = np.where(np.array(prediction) == output)[-1]
output = prediction.iloc[0:,where].columns[0]
print ('My Prediction is {}'.format(output))
for i in prediction.columns:
    if i != 'Soccer Ball':
        print ('{}:\t\t{}%'.format(i, prediction[i].iloc[-1]*100))
    else:
        print ('{}:\t{}%'.format(i, prediction[i].iloc[-1]*100))
Ai = Ai.append(X_test)
actual_name = input('What Was the Drawing?')
Ai['Label'].iloc[-1] = (actual_name)
Ai['Label'].iloc[-1:] = Ai['Label'].iloc[-1:].map({'Book': 0, 'Lamp': 1, 'Fork': 2, 'Phone': 3, 'Tv': 4, 'Tree': 5, 
                                                   'Apple': 6, 'Soccer Ball': 7, 'Car': 8, 'Mug': 9, 'Outlet': 10,
                                                   'Clock': 11, 'Door': 12, 'Heart':13})
save = input('Save?').lower()
if 'y' in save:
    compression_opts = dict(method='infer',
                        archive_name='PictionaryAi.csv')
    Ai.to_csv('PictionaryAi.csv', index=False,
          compression=compression_opts)
# Ai makes a guess
# If Ai doesnt guess before the time ends then go to lose screen
# Otherwise go to win screen