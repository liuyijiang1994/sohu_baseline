import re
import json
import random
import args
import os
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from parser import entity_extractor
import torch

random.seed(233)
tokenizer = BasicTokenizer()
test_data = 'data/test_data.pth'


class doc:
    def __init__(self, news_id, content):
        self.id = news_id
        self.title = content[0]
        self.content = content[1:]


def make_submit(doc, parser: entity_extractor):
    entity_cont = {}
    title_entity = parser.extract([doc.title])
    content_entity = parser.extract(doc.content)
    doc_entity = content_entity | title_entity
    for entity in doc_entity:
        if doc.title.find(entity) > -1:
            entity_cont[entity] = entity_cont.get(entity, 0) + 10
        for stc in doc.content:
            if stc.find(entity) > -1:
                entity_cont[entity] = entity_cont.get(entity, 0) + 1
    data = {'id': doc.id, 'title_entity': list(title_entity), 'entity_cont': entity_cont}
    return json.dumps(data)


def load_docs(data_file):
    doc_list = []
    k = 0
    if os.path.exists(test_data):
        return torch.load(test_data)
    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            k += 1
            if k % 100 == 0:
                print(k)

            data = json.loads(line.strip())
            news_id = data['newsId']
            title = data['title']
            content = data['content']

            sentences = [title]
            for seq in re.split(r'[\n。？！?!]', content):
                seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
                sentences.append(seq)

            sentences = [''.join(tokenizer.tokenize(seq)[:args.max_seq_length]).strip() for seq in sentences]
            doc_list.append(doc(news_id, sentences))
    torch.save(doc_list, test_data)
    return doc_list


if __name__ == '__main__':
    ee = entity_extractor()
    doc_list = load_docs('data/coreEntityEmotion_test_stage1.txt')
    with open('output/entity_result.txt', 'w') as w:
        for ix, doc in enumerate(doc_list):
            j = make_submit(doc, ee)
            w.write(j + '\n')
            if ix % 100 == 0:
                print(ix)
