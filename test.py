import os
from net.bert_ner import Bert_CRF
from data_loader import create_inference_iter
import torch

import args

from util.model_util import load_model

from util.score import get_tags_BIESO


def start():
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print('create_iter')
    eval_iter = create_inference_iter()
    print('create_iter finished')
    # ------------------判断CUDA模式----------------------
    with torch.no_grad():
        for step, batch_data in enumerate(eval_iter):
            text_list, batch = batch_data
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            print(text_list)


if __name__ == '__main__':
    start()
