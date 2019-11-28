# Train Config

learning_rate = 5e-5
min_learning_rate = 1e-6
config_path = './ckpt/albert_config_xlarge.json'
checkpoint_path = './ckpt/albert_model.ckpt'
dict_path = './ckpt/vocab.txt'
MAX_LEN = 128
TITLE_MAX_LEN = 64
CONTENT_MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 64
FOLDS_SPLIT = 6

model_save_path = './model_save/bert2_pred/'
