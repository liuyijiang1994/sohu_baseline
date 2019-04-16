# -----------ARGS---------------------
ROOT_DIR = "/home/liuyijiang/parcharm2/Bert_Chinese_Ner_pytorch/"
RAW_SOURCE_DATA = "data/source.txt"
RAW_TARGET_DATA = "data/target.txt"

ENTITY_TAG_DATA = 'data/entity_tag.txt'
NO_ENTITY_TAG_DATA = 'data/no_entity_tag.txt'

STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
VOCAB_FILE = "model/vocab.txt"
TRAIN = "data/train.json"
VALID = "data/dev.json"
log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "model/"
output_dir = "output/checkpoint"  # checkpoint和预测输出文件夹

entity_torch_data = 'data/entity_torch.pth'
no_entity_torch_data = 'data/no_entity_torch.pth'

bert_model = "bert-base-chinese"  # BERT 预训练模型种类 bert-base-chinese
task_name = "bert_ner"  # 训练任务名称
bert_cache = '/cache/'
flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 220
do_lower_case = True
train_batch_size = 16
eval_batch_size = 16
inference_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 20
warmup_proportion = 0.1
no_cuda = False
seed = 233
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = ["S-entity", "B-entity", "I-entity", "E-entity", "O"]
device = "cuda"
train_size = 0.7
