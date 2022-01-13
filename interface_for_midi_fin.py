import streamlit as st

import pretty_midi as pm
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
st.write("""
# Сеть генерирующая мелодию в стиле House
""")

def midi_to_list(midi):
    pitch_list = []
    for note in midi.instruments[0].notes:
        pitch_list.append(note.pitch)
    return pitch_list

def list_shift(pitch_list, lowest, highest):
    pitch_list_new = []
    for note in pitch_list:
        if(note > highest):
            print('note with pitch ' + str(note) + 'spotted')
        pitch_list_new.append(note - lowest)
    return pitch_list_new

notes_4_first_opt = {0: 0, 2: 1, 3:2, 5:3, 7:4, 8:5, 10:6, 12:7}

def first_opt(pitch_list_new):
    tmp = []
    for note in pitch_list_new:
        tmp.append(notes_4_first_opt[note])
    return tmp

to_pitch_dict = {0: 0, 1: 2, 2:3, 3:5, 4:7, 5:8, 6:10, 7:12}


def to_pitch(final_midi):
    tmp = []
    for note in final_midi:
        tmp.append(to_pitch_dict[note])
    return tmp

def shift_back(final_midi, lowest):
    tmp = []
    for note in final_midi:
        tmp.append(note + lowest)
    return tmp

#создание нейросети 1 и загрузка весов 1

model_1 = Sequential()
model_1.add(LSTM(512, input_shape=(5, 8), return_sequences=True))
model_1.add(Dropout(0.2))
model_1.add(LSTM(256, return_sequences=True))
model_1.add(Dropout(0.2))
model_1.add(LSTM(128))
model_1.add(Dropout(0.2))
model_1.add(Dense(8, activation='softmax'))

filename = "model_1_weights_saved.hdf5" 
model_1.load_weights(filename) 
model_1.compile(loss='categorical_crossentropy', optimizer='adam')

#создание нейросети 1 и загрузка весов 1

model_2 = Sequential()
model_2.add(LSTM(512, input_shape=(5, 14), return_sequences=True))
model_2.add(Dropout(0.2))
model_2.add(LSTM(256, return_sequences=True))
model_2.add(Dropout(0.2))
model_2.add(LSTM(128))
model_2.add(Dropout(0.2))
model_2.add(Dense(14, activation='softmax'))

filename = "model_2_weights_saved.hdf5" 
model_2.load_weights(filename) 
model_2.compile(loss='categorical_crossentropy', optimizer='adam')
   
clicked = st.button("Создать мелодии")

if clicked:

    #генерация мелодии 1

    melody_pred = np.zeros((5))
    X_pred = np.zeros((1,5,8))

    melody_pred[0] = int(np.round(7*np.random.rand()))
    melody_pred[1] = int(np.round(7*np.random.rand()))
    melody_pred[2] = int(np.round(7*np.random.rand()))
    melody_pred[3] = int(np.round(7*np.random.rand()))
    melody_pred[4] = int(np.round(7*np.random.rand()))

    for i in range(melody_pred.shape[0]):
        X_pred[0][i][int(melody_pred[i])] = 1
        
    X_pred_012 = X_pred

    final_midi = []
    zeros = np.zeros((8))
    for i in range(0,19):
        prediction = np.argmax(model_1.predict(X_pred, verbose = 0))
        
        final_midi.append(prediction)
        X_pred[0][0] = X_pred[0][1]
        X_pred[0][1] = X_pred[0][2]
        X_pred[0][2] = X_pred[0][3]
        X_pred[0][3] = X_pred[0][4]
        X_pred[0][4] = zeros
        X_pred[0][4][prediction] = 1

    melody_start = []
    for note in melody_pred:
        melody_start.append(int(note))
    final_midi = melody_start + final_midi

    final_shifted = shift_back(to_pitch(final_midi), 69)

    export_midi = pm.PrettyMIDI('Dataset/1.mid')

    for i in range(24):
        export_midi.instruments[0].notes[i].pitch = final_shifted[i]

    export_midi.write('Output_1.mid')

    #генерация мелодии 2

    melody_pred = np.zeros((5))
    X_pred = np.zeros((1,5,14))

    melody_pred[0] = int(np.round(13*np.random.rand()))
    melody_pred[1] = int(np.round(13*np.random.rand()))
    melody_pred[2] = int(np.round(13*np.random.rand()))
    melody_pred[3] = int(np.round(13*np.random.rand()))
    melody_pred[4] = int(np.round(13*np.random.rand()))

    for i in range(melody_pred.shape[0]):
        X_pred[0][i][int(melody_pred[i])] = 1
        
    X_pred_012 = X_pred

    final_midi = []
    zeros = np.zeros((14))
    for i in range(0,19):
        prediction = np.argmax(model_2.predict(X_pred, verbose = 0))
        
        final_midi.append(prediction)
        X_pred[0][0] = X_pred[0][1]
        X_pred[0][1] = X_pred[0][2]
        X_pred[0][2] = X_pred[0][3]
        X_pred[0][3] = X_pred[0][4]
        X_pred[0][4] = zeros
        X_pred[0][4][prediction] = 1

    melody_start = []
    for note in melody_pred:
        melody_start.append(int(note))
    final_midi = melody_start + final_midi

    final_shifted = shift_back(final_midi, 69)

    export_midi = pm.PrettyMIDI('Dataset/1.mid')

    for i in range(24):
        export_midi.instruments[0].notes[i].pitch = final_shifted[i]

    export_midi.write('Output_2.mid')

st.write("""
Создание мелодии по одной тональности и 8 нотам
""")

# кнопка скачивания мелодии 1

with open("Output_1.mid", "rb") as file:
    dwnldbtn = st.download_button("Скачать midi файл 1", data=file, file_name='Output_1.mid', mime=None, key=None, help=None, on_click=None, args=None, kwargs=None)
    if file is None:
        st.error("Файл еще не создан")
        raise ValueError("Файл еще не создан")
# интерфейс для второй мелодии(пока заглушка)

st.write("""
Мелодии по 2 тональностям с большим количеством нот
""")

with open("Output_2.mid", "rb") as file2:
    dwnldbtn2 = st.download_button("Скачать midi файл 2", data=file2, file_name='Output_2.mid', mime=None, key=None, help=None, on_click=None, args=None, kwargs=None)
    if file2 is None:
        st.error("Файл еще не создан")
        raise ValueError("Файл еще не создан")