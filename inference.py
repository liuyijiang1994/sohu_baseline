import os
from net.bert_ner import Bert_CRF
from data_loader import create_inference_iter
import torch

import args

from util.model_util import load_model

from util.score import get_tags_BIESO


def start():
    # produce_data()
    model = Bert_CRF()
    model.load_state_dict(load_model(args.output_dir))
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    print('create_iter')
    eval_iter = create_inference_iter()
    print('create_iter finished')
    # ------------------判断CUDA模式----------------------
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # -----------------------验证----------------------------
    model.eval()
    count = 0
    eval_loss, eval_acc, eval_f1 = 0, 0, 0
    with torch.no_grad():
        for step, batch_data in enumerate(eval_iter):
            text_list, batch = batch_data
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            eval_los = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
            eval_loss = eval_los + eval_loss
            count += 1
            predicts = model.predict(bert_encode, output_mask)
            for ix, predict in enumerate(predicts):
                text = text_list[ix]
                label = label_ids[ix]
                predict = predict[predict != -1]
                label = label[label != -1]
                pre_tags = get_tags_BIESO(predict.cpu().numpy().tolist())
                label_tags = get_tags_BIESO(label.cpu().numpy().tolist())
                pre_entities = [text[tag[0]:tag[1] + 1] for tag in pre_tags]
                label_entities = [text[tag[0]:tag[1] + 1] for tag in label_tags]
                print(input_ids[ix])
                print(f'pre:{pre_entities}')
                print(f'label:{label_entities}')
                print(text)
                print('=' * 10)


if __name__ == '__main__':
    start()
