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
oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)

logger.info("===================== PREDICT =====================")

for fold in range(FOLDS_SPLIT):
    logger.info('================     fold {}        ==============='.format(fold))
    model = get_model()
    model.load_model('./model_save/bert_model{}.h5'.format(fold))
    oof_test += predict(model, [test_achievements, test_requirements])
    K.clear_session()

oof_test /= 5
np.savetxt('./model_save/test_bert.txt', oof_test)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('./bert_{}.csv'.format(cv_score), index=False)