import re
import json
import args
import random

from pytorch_pretrained_bert.tokenization import BertTokenizer

random.seed(233)


def load_data(data_file):
    stc_list = []
    label_list = []

    with open('data/entity.txt', 'w', encoding='utf-8') as w:
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:

                data = json.loads(line.strip())
                news_id = data['newsId']
                if 'coreEntityEmotions' in data:
                    entities = data['coreEntityEmotions']
                else:
                    entities = []
                title = data['title']
                content = data['content']

                sentences = [title]
                for seq in re.split(r'[\n。？！?!]', content):
                    seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
                    if len(seq) <= 220:
                        sentences.append(seq)

                for seq in sentences:
                    # find entities
                    seq = ''.join(tokenizer.tokenize(seq))
                    ent_spans = []
                    ent_emotions = []
                    for ent_emo in entities:
                        ent = ent_emo['entity']
                        emo = ent_emo['emotion']

                        last_idx = 0
                        while True:
                            if last_idx >= len(seq):
                                break
                            start = seq[last_idx:].find(ent)
                            if start == -1:
                                break
                            end = start + len(ent)
                            ent_spans.append((start + last_idx, end + last_idx))
                            ent_emotions.append(emo)
                            last_idx = end + last_idx

                    sub_ent = []
                    sub_emo = []
                    sub_c = []
                    for i, c in enumerate(seq):
                        sub_c.append(c)
                        ent_tag = 'O'
                        emo_tag = 'EMPTY'
                        for em, sp in zip(ent_emotions, ent_spans):
                            if sp[0] == i:
                                if sp[0] == sp[1]:
                                    ent_tag = 'S-entity'
                                else:
                                    ent_tag = 'B-entity'
                                emo_tag = em
                                break
                            elif sp[1] - 1 == i:
                                ent_tag = 'E-entity'
                                emo_tag = em
                                break
                            elif sp[0] < i < sp[1] - 1:
                                ent_tag = 'I-entity'
                                emo_tag = em
                                break
                        sub_ent.append(ent_tag)
                        sub_emo.append(emo_tag)
                    assert len(seq) == len(sub_ent)

                    # 将'[UNK]'合起来标一个'O'
                    source = ''.join(seq)
                    target = sub_ent
                    new_source = []
                    new_target = []
                    is_unk = 0
                    if '[UNK]' in source:
                        is_unk = 1
                        unk_index_lst = [i.start() for i in re.finditer('\[UNK\]', source)]
                    i = 0
                    while i < len(target):
                        if is_unk and i in unk_index_lst:
                            new_source.append('[UNK]')
                            new_target.append('O')
                            i += 5
                        else:
                            new_source.append(source[i])
                            new_target.append(target[i])
                            i += 1
                    assert len(new_source) == len(new_target)

                    # source = ' '.join(new_source)
                    # target = ' '.join(new_target)
                    # source_file.write(source + '\n')
                    # target_file.write(target + '\n')

                    stc_list.append(new_source)
                    label_list.append(new_target)

                    # for c, en, em in zip(seq, sub_ent, sub_emo):
                    #     stc.append(c)
                    #     label.append(en)
                    # stc.append('')

        print(f'stc:{len(stc_list)}')
        print(f'label:{len(label_list)}')

        c = list(zip(stc_list, label_list))
        random.shuffle(c)
        stc_list[:], label_list[:] = zip(*c)
        source_file = open('data/source.txt', 'w', encoding='utf-8')
        target_file = open('data/target.txt', 'w', encoding='utf-8')

        source_file.writelines([' '.join(source) + '\n' for source in stc_list])
        target_file.writelines([' '.join(source) + '\n' for source in label_list])

        source_file.close()
        target_file.close()

        # len_cont = {}  # label.append('')
        # for stc, label in zip(stc_list, label_list):
        #     len_cont[len(stc)] = len_cont.get(len(stc), 0) + 1
        #     if len(stc) <= 220:
        #         out = [f'{s}\t{l}\n' for s, l in zip(stc, label)]
        #         w.writelines(out)
        #         w.write('\n')
        # len_cont = sorted(len_cont.items(), key=lambda x: x[0])
        # print(len_cont)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # print(tokenizer.tokenize('你好   你好   好好好'))
    data_file = 'data/coreEntityEmotion_train.txt'

    # data_file = sys.argv[1]
    load_data(data_file)
