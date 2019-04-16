import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pytorch_pretrained_bert.modeling import BertForTokenClassification
import args
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class Bert_CRF(nn.Module):
    def __init__(self):
        super(Bert_CRF, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                                               num_labels=len(args.labels))

        self.crf = CRF(len(args.labels))

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None,
                output_all_encoded_layers=False):
        logit = self.bert(input_ids, token_type_ids)
        return logit

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
