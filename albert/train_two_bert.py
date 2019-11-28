import json
import numpy as np
import time
import logging
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from keras_bert import Tokenizer
# from keras_bert import load_trained_model_from_checkpoint Tokenizer
# from bert4keras.utils import Tokenizer
from bert4keras.bert import build_bert_model
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from keras.utils import multi_gpu_model
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

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

def read_data(file_path, id, name):
    train_id = []
    train_title = []
    train_text = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(',')  # split the csv file [id, title, text]
            train_id.append(line[0].replace('\'', '').replace(' ', ''))
            train_title.append(line[1].replace('\t','').replace('\'',''))
            train_text.append('，'.join(line[2:]))
    output = pd.DataFrame(dtype=str)
    output[id] = train_id
    output[name + '_title'] = train_title
    output[name + '_content'] = train_text
    return output

# read the data 
train_interrelation = pd.read_csv('./input/Train_Interrelation.csv', dtype=str)
# print("train_interralation:\t",len(train_interrelation))
Train_Achievements = read_data('./input/Train_Achievements.csv', 'Aid', 'Achievements')
# print("Train_Achievements:\t",len(Train_Achievements))
Requirements = read_data('./input/Requirements.csv', 'Rid', 'Requirements')
# print("Requents:\t",len(Requirements))
TestPrediction = pd.read_csv('./input/TestPrediction.csv', dtype=str)
# print("Test_Prediction:\t",len(TestPrediction))
Test_Achievements = read_data('./input/Test_Achievements.csv', 'Aid', 'Achievements')

Train_Achievements_Summary = pd.read_csv('./input/Train_Achivements_Summary.csv')
Requirements_Summary = pd.read_csv('./input/Train_Requirements_Summary.csv')
Test_Achievements_Summary = pd.read_csv('./input/Test_Achivements_Summary.csv')

Train_Achievements_Summary = Train_Achievements_Summary.set_index("Aid")
Requirements_Summary = Requirements_Summary.set_index("Rid")
Test_Achievements_Summary = Test_Achievements_Summary.set_index("Aid")

# merge the dataframe
train = pd.merge(train_interrelation, Train_Achievements, on='Aid', how='left')
train = pd.merge(train, Requirements, on='Rid', how='left')

def add_train_summary(row):
    row["Achivements_summary_128"] = Train_Achievements_Summary.loc['\'' + row["Aid"] + '\'' + ' ']["Achivements_summary_128"]
    if type(row["Achivements_summary_128"]) == str: 
        row["Achivements_summary_128"] = row["Achivements_summary_128"].replace('\t','').replace('\'','').replace(' ','').replace('\\n',' ')
    row["Requirements_summary_128"] = Requirements_Summary.loc['\''+row["Rid"]+'\'' +  ' ']["Requirements_summary_128"]
    if type(row["Requirements_summary_128"]) == str:
        row["Requirements_summary_128"] = row["Requirements_summary_128"].replace('\t','').replace('\'','').replace(' ','').replace('\\n',' ')
    return row

def add_test_summary(row):
    row["Achivements_summary_128"] = Test_Achievements_Summary.loc['\'' + row["Aid"] + '\'' + ' ']["Achivements_summary_128"]
    if type(row["Achivements_summary_128"]) == str:
        row["Achivements_summary_128"] = row["Achivements_summary_128"].replace('\t','').replace('\'','').replace(' ','').replace('\\n',' ')
    row["Requirements_summary_128"] = Requirements_Summary.loc['\''+row["Rid"]+'\'' +  ' ']["Requirements_summary_128"]
    if type(row["Requirements_summary_128"]) == str:
        row["Requirements_summary_128"] = row["Requirements_summary_128"].replace('\t','').replace('\'','').replace(' ','').replace('\\n',' ')
    return row

train = train.apply(add_train_summary, axis=1)

test = pd.merge(TestPrediction, Test_Achievements, on='Aid', how='left')
test = pd.merge(test, Requirements, on='Rid', how='left')
test = test.apply(add_test_summary, axis=1)


train_achievements_title = train['Achievements_title'].values
train_requirements_title = train['Requirements_title'].values

train_achievements_content = train['Achivements_summary_128'].values
train_requirements_content = train['Requirements_summary_128'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements_title = test['Achievements_title'].values
test_requirements_title = test['Requirements_title'].values

test_achievements_content = test['Achivements_summary_128'].values
test_requirements_content = test['Requirements_summary_128'].values

# data generator

class data_generator:
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, X3, X4, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T1, T1_, T2, T2_, Y = [], [], [], [], []
            for c, i in enumerate(idxs):
                achievements_title = X1[i]
                requirements_title = X2[i]
                achievements_content = X3[i]
                requirements_content = X4[i]
                t1, t1_ = tokenizer.encode(first=achievements_title, second=requirements_title, max_len=TITLE_MAX_LEN)
                if type(achievements_content) != str:
                    achievements_content = achievements_title
                if type(requirements_content) != str:
                    requirements_content = requirements
                t2, t2_ = tokenizer.encode(first=achievements_content, second=requirements_content, max_len=CONTENT_MAX_LEN)
                T1.append(t1)
                T1_.append(t1_)
                T2.append(t2)
                T2_.append(t2_)
                Y.append(y[i])
                if len(T1) == self.batch_size or i == idxs[-1]:
                    T1 = np.array(T1)
                    T1_ = np.array(T1_)
                    T2 = np.array(T2)
                    T2_ = np.array(T2_)
                    Y = np.array(Y)
                    yield [T1, T1_, T2, T2_], Y
                    T1, T1_, T2, T2_, Y = [], [], [], [], []

def get_model():
    bert_model1 = build_bert_model(
        config_path,
        checkpoint_path,
        albert=True
    )
    bert_model2 = build_bert_model(
        config_path,
        checkpoint_path,
        albert=True
    )
    for l in bert_model1.layers:
        l.trainable = True
    for l in bert_model2.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T3 = Input(shape=(None,))
    T4 = Input(shape=(None,))

    B1 = bert_model1([T1, T2])
    B2 = bert_model2([T3, T4])

    B1 = Lambda(lambda x: x[:, 0])(B1)
    B2 = Lambda(lambda x: x[:, 0])(B2)

    R = concatenate([B1, B2], axis=1)

    output = Dense(4, activation='softmax')(R)

    model = Model([T1, T2, T3, T4], output)

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return parallel_model, model

class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save(model_save_path+'bert_model{}.h5'.format(fold))
            model.save_weights(model_save_path+'bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_x3, val_x4, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements_title = val_x1[i]
            requirements_title = val_x2[i]
            achievements_content = val_x3[i]
            requirements_content = val_x4[i]
            if type(achievements_content) != str:
                achievements_content = achievements_title
            if type(requirements_content) != str:
                requirements_content = requirements
            t1, t1_ = tokenizer.encode(first=achievements_title, second=requirements_title, max_len=TITLE_MAX_LEN)
            t2, t2_ = tokenizer.encode(first=achievements_content, second=requirements_content, max_len=CONTENT_MAX_LEN)
            T1, T1_ = np.array([t1]), np.array([t1_])
            T2, T2_ = np.array([t2]), np.array([t2_])
            _prob = model.predict([T1, T1_, T2, T2_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0]+1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y+1, self.predict))
        acc = accuracy_score(val_y+1, self.predict)
        f1 = f1_score(val_y+1, self.predict, average='macro')
        return score, acc, f1

skf = StratifiedKFold(n_splits=FOLDS_SPLIT, shuffle=True, random_state=67)

def predict(data):
    prob = []
    val_x1, val_x2, val_x3, val_x4 = data
    for i in tqdm(range(len(val_x1))):
        achievements_title = val_x1[i]
        requirements_title = val_x2[i]
        achievements_content = val_x3[i]
        requirements_content = val_x4[i]

        if type(achievements_content) != str:
            achievements_content = achievements_title
        if type(requirements_content) != str:
            requirements_content = requirements

        t1, t1_ = tokenizer.encode(first=achievements_title, second=requirements_title, max_len=TITLE_MAX_LEN)
        t2, t2_ = tokenizer.encode(first=achievements_content, second=requirements_content, max_len=CONTENT_MAX_LEN)

        T1, T1_ = np.array([t1]), np.array([t1_])
        T2, T2_ = np.array([t2]), np.array([t2_])
        _prob = model.predict([T1, T1_, T2, T2_])
        prob.append(_prob[0])
    return prob


# Train 
oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements_title, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_achievements_title[train_index]
    x2 = train_requirements_title[train_index]
    x3 = train_achievements_content[train_index]
    x4 = train_requirements_content[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements_title[valid_index]
    val_x2 = train_requirements_title[valid_index]
    val_x3 = train_achievements_content[valid_index]
    val_x4 = train_requirements_content[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, x2, x3, x4, y])
    evaluator = Evaluate([val_x1, val_x2, val_x3, val_x4, val_y, val_cat], valid_index)

    parallel_model,model = get_model()
    parallel_model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=EPOCHS,
                        callbacks=[evaluator]
                       )
    model.load_weights(model_save_path+'bert{}.w'.format(fold))
    oof_test += predict([test_achievements_title, test_requirements_title, test_achievements_content, test_requirements_content])
    K.clear_session()


oof_test /= 5
np.savetxt('./model_save/train_bert_bert2_fillnan.txt', oof_train)
np.savetxt('./model_save/test_bert_bert2_fillnan.txt', oof_test)

cv_score = 1.0 / (1 + mean_absolute_error(labels+1, np.argmax(oof_train, axis=1) + 1))
print("Cv_Score",cv_score)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('./bert_{}.csv'.format(cv_score), index=False)
