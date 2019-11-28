# Train Config

learning_rate = 5e-5
min_learning_rate = 5e-6
config_path = './ckpt/albert_config_xlarge.json'
checkpoint_path = './ckpt/albert_model.ckpt'
dict_path = './ckpt/vocab.txt'
MAX_LEN = 64
TITLE_MAX_LEN = 64
CONTENT_MAX_LEN = 128
EPOCHS = 10
BATCH_SIZE = 64
FOLDS_SPLIT = 5

model_save_path = './model_save/bert2_clean/'
