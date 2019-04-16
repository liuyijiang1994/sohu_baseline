import os
from net.bert_ner import Bert_CRF
from data_loader import create_batch_iter
import torch

import args

from util.model_util import save_model, load_model

from util.score import score_predict


def start():
    # produce_data()
    model = Bert_CRF()
    model.load_state_dict(load_model(args.output_dir))
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    print('create_iter')
    eval_iter = create_batch_iter("valid")
    print('create_iter finished')
    # ------------------判断CUDA模式----------------------
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # -----------------------验证----------------------------
    model.eval()
    count = 0
    y_predicts, y_labels = [], []
    eval_loss, eval_acc, eval_f1 = 0, 0, 0
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            eval_los = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
            eval_loss = eval_los + eval_loss
            count += 1
            predicts = model.predict(bert_encode, output_mask)
            predict_tensor = predicts.cpu()
            label_tensor = label_ids.cpu()
            y_predicts.append(predicts)
            y_labels.append(label_ids)
            entity_precision, entity_recall, entity_f1 = score_predict(label_tensor, predict_tensor)
            print('\n step :%d - eval_loss: %4f - ent_p:%4f - ent_r:%4f - ent_f1:%4f\n'
                  % (step, eval_loss.item() / count, entity_precision, entity_recall, entity_f1))

            label_ids = label_ids.view(1, -1).squeeze()
            predicts = predicts.view(1, -1).squeeze()
            label_ids = label_ids[label_ids != -1]
            predicts = predicts[predicts != -1]
            assert len(label_ids) == len(predicts)

        eval_predicted = torch.cat(y_predicts, dim=0).cpu()
        eval_labeled = torch.cat(y_labels, dim=0).cpu()
        entity_precision, entity_recall, entity_f1 = score_predict(eval_labeled, eval_predicted)
        print('\n\n- eval_loss: %4f - eval_acc:%4f - eval_f1:%4f - ent_p:%4f - ent_r:%4f - ent_f1:%4f\n'
              % (eval_loss.item() / count, eval_acc, eval_f1, entity_precision, entity_recall, entity_f1))


if __name__ == '__main__':
    start()
