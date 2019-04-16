import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_processor import MyPro, convert_examples_to_features
import args
from util.Logginger import init_logger
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from mydataset import MyTensorDataset

logger = init_logger("bert_ner", logging_path=args.log_path)


def init_params():
    processors = {"bert_ner": MyPro}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                              do_lower_case=args.do_lower_case)
    return processor, tokenizer


def create_inference_iter():
    processor, tokenizer = init_params()
    examples = processor.get_valid_examples()[:100]
    batch_size = args.inference_batch_size
    label_list = processor.get_labels()
    # 特征
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)
    all_text = [''.join(example.text_a) for example in examples]
    print(all_text)
    # 数据集
    data = MyTensorDataset(all_text, all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)
    sampler = SequentialSampler(data)
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return iterator


def create_batch_iter(mode):
    """构造迭代器"""
    processor, tokenizer = init_params()
    if mode == "train":
        examples = processor.get_train_examples()

        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = args.train_batch_size

        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "valid":
        examples = processor.get_valid_examples()
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()

    # 特征
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "valid":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "valid":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)
