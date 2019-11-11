import json
import numpy as np
import time
import logging
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from tqdm import tqdm
from config import *
from keras.models import load_model

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

def read_data(file_path, id, name):
    train_id = []
    train_title = []
    train_text = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(',')  # split the csv file [id, title, text]
            train_id.append(line[0].replace('\'', '').replace(' ', ''))
            train_title.append(line[1])
            train_text.append('，'.join(line[2:]))
    output = pd.DataFrame(dtype=str)
    output[id] = train_id
    output[name + '_title'] = train_title
    output[name + '_content'] = train_text
    return output

# read the data 
train_interrelation = pd.read_csv('./input/Train_Interrelation.csv', dtype=str)
print("train_interralation:\t",len(train_interrelation))
Train_Achievements = read_data('./input/Train_Achievements.csv', 'Aid', 'Achievements')
print("Train_Achievements:\t",len(Train_Achievements))
Requirements = read_data('./input/Requirements.csv', 'Rid', 'Requirements')
print("Requents:\t",len(Requirements))
TestPrediction = pd.read_csv('./input/TestPrediction.csv', dtype=str)
print("Test_Prediction:\t",len(TestPrediction))
Test_Achievements = read_data('./input/Test_Achievements.csv', 'Aid', 'Achievements')

# merge the dataframe
train = pd.merge(train_interrelation, Train_Achievements, on='Aid', how='left')
train = pd.merge(train, Requirements, on='Rid', how='left')

test = pd.merge(TestPrediction, Test_Achievements, on='Aid', how='left')
test = pd.merge(test, Requirements, on='Rid', how='left')

train_achievements = train['Achievements_title'].values
train_requirements = train['Requirements_title'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements = test['Achievements_title'].values
test_requirements = test['Requirements_title'].values


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(4, activation='softmax')(T)

    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

def predict(pred_model, data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]

        t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = pred_model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob

# predict
oof_test = np.zeros((len(test), 4), dtype=np.float32)

print("===================== PREDICT =====================")

for fold in range(4):
    print('================     fold {}        ==============='.format(fold))
    model = get_model()
    model.load_weights('./model_save/bert{}.h5'.format(fold))
    oof_test += predict(model, [test_achievements, test_requirements])
    K.clear_session()

oof_test /= 4
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('./bert_.csv', index=False)
