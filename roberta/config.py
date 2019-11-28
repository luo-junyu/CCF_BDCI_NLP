# Train Config

learning_rate = 5e-5
min_learning_rate = 5e-6
config_path = './roberta-large/bert_config_large.json'
checkpoint_path = './roberta-large/roberta_zh_large_model.ckpt'
dict_path = './roberta-large/vocab.txt'
MAX_LEN = 64
TITLE_MAX_LEN = 64
CONTENT_MAX_LEN = 128
EPOCHS = 40
BATCH_SIZE = 256
FOLDS_SPLIT = 5

model_save_path = './model_save/1Bert_finetune_30/'
