import re
import json
import matplotlib.pyplot as plt
import random

random.seed(233)


def split_fn(re_str, text):
    substcs = re.split(re_str, text)
    new_sub = []

    for i in range(0, len(substcs), 2):
        if i + 1 >= len(substcs):
            new_sub.append(substcs[i])
        else:
            new_sub.append(substcs[i] + substcs[i + 1])
    return new_sub


def split_sub_stc(text):
    re_str = r"([,，])"
    return split_fn(re_str, text)


# 以句号等分割句子
def split_stc(text):
    re_str = r"([。!！?？；；;\n])"
    new_sub = split_fn(re_str, text)
    new_sub = [sub.replace('\n', '') for sub in new_sub if sub.replace('\n', '').strip() != '']
    return new_sub


def source_to_target(source, entity_list):
    target = ['O'] * len(source)
    for entity in entity_list:
        begin = 0
        while len(entity) > 0 and source.find(entity, begin) >= 0:
            tbegin = source.find(entity, begin)
            target[tbegin] = 'B'
            for i in range(len(entity) - 1):
                target[tbegin + 1 + i] = 'I'
            begin = tbegin + 1
    return ''.join(target)


def has_entity(content, entity_list):
    for e in entity_list:
        if content.find(e) >= 0:
            return True
    return False


source_list = []
target_list = []

# a = '后来，她在采访中提到了那段经历，坦言当时自己太天真，就觉得女人不用太拼，找个男人靠一靠，小鸟依人过一生就挺好。'
# b = ['后来', '坦言当时自己']
# print(source_to_target(a, b))

seq_len_cont = {}
has_entity_cont = 0
with open('data/coreEntityEmotion_train.txt', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        entity_emotions = item['coreEntityEmotions']
        entities = [ee['entity'] for ee in entity_emotions]
        title = item["title"].strip()
        content_list = split_stc(item["content"].strip())
        source_list.append(title)
        target_list.append(source_to_target(title, entities))
        for content in content_list:
            if (len(content) > 8 and len(content) < 160) or has_entity(content, entities):
                if len(content) > 160:
                    c_list = split_sub_stc(content)
                    for c in c_list:
                        if (len(c) > 8 and len(c) < 160) or has_entity(c, entities):
                            if len(c) > 160:
                                c = c[:160]
                            source_list.append(c)
                            target_list.append(source_to_target(c, entities))
                            seq_len_cont[len(c)] = seq_len_cont.get(len(c), 0) + 1
                            if has_entity(c, entities):
                                has_entity_cont += 1

                else:
                    source_list.append(content)
                    target_list.append(source_to_target(content, entities))
                    seq_len_cont[len(content)] = seq_len_cont.get(len(content), 0) + 1
                    if has_entity(content, entities):
                        has_entity_cont += 1

    print(len(source_list))
    print(len(target_list))
    print(has_entity_cont)
p1 = []
p2 = []
for k, v in seq_len_cont.items():
    p1.append(k)
    p2.append(v)
plt.figure('Draw')
plt.scatter(p1, p2)  # plot绘制折线图
plt.draw()  # 显示绘图
plt.pause(5)  # 显示5秒
plt.savefig("easyplot01.jpg")  # 保存图象
plt.close()  # 关闭图表
seq_len_cont = sorted(seq_len_cont.items(), key=lambda x: x[0])


# for k, v in seq_len_cont:
#     print(k, v)


def save_data(path, data_list):
    with open(path, 'w') as w:
        for stc, label in data_list:
            for c, l in zip(stc, label):
                w.write(f'{c}\t{l}\n')
            w.write('\n')


train_len = int(0.5 * len(source_list))
valid_len = int(0.8 * len(source_list))
c = list(zip(source_list, target_list))
random.shuffle(c)

train_data = c[:]
save_data('data/train_bio.txt', train_data)
#
# save_data('data/train.txt', train_data)
# save_data('data/valid.txt', valid_data)
# save_data('data/test.txt', test_data)
