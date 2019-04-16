import torch

BIO_tag_map = {'B-entity': 0, 'I-entity': 1, 'O': 2}
BIEO_tag_map = {'B-entity': 0, 'I-entity': 1, 'E-entity': 2, 'O': 3}
BIESO_tag_map = {'S-entity': 0, 'B-entity': 1, 'I-entity': 2, 'E-entity': 3, 'O': 4}


def get_tags_BIO(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = -1
    for index, tag in enumerate(path):
        if tag == begin_tag:
            if index == len(path) - 1:
                end = index
                tags.append([index, end])
            if last_tag in [mid_tag, begin_tag] and begin > -1:
                end = index - 1
                tags.append([begin, end])
            begin = index
        elif tag == mid_tag and last_tag in [mid_tag, begin_tag] and begin > -1 and index == len(path) - 1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index - 1
            tags.append([begin, end])
            begin = -1
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags


def get_tags_BIEO(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = -1
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            if index == len(path) - 1:
                end = index
                tags.append([index, end])
            if last_tag == begin_tag:
                end = index - 1
                tags.append([begin, end])
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag and last_tag == begin_tag:
            end = index - 1
            tags.append([begin, end])
            begin = -1
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags


def get_tags_BIESO(path, tag='entity', tag_map=BIESO_tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S-" + tag)
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = -1
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == single_tag:
            tags.append([index, index])
            begin = -1
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags


def f1_score(tar_path, pre_path, tag, type):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        if type == "BIO":
            tar_tags = get_tags_BIO(tar, tag, BIO_tag_map)
            pre_tags = get_tags_BIO(pre, tag, BIO_tag_map)
        elif type == "BIEO":
            tar_tags = get_tags_BIEO(tar, tag, BIEO_tag_map)
            pre_tags = get_tags_BIEO(pre, tag, BIEO_tag_map)
        elif type == "BIESO":
            tar_tags = get_tags_BIESO(tar, tag, BIESO_tag_map)
            pre_tags = get_tags_BIESO(pre, tag, BIESO_tag_map)
        # print(tar_tags)
        # print(pre_tags)
        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1
    # print(right)
    # print(origin)
    # print(found)
    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def score_predict(label_tensor, predict_tensor, lable_type='BIESO'):
    predict_list = []
    label_list = []
    tag = 'entity'
    for x in range(label_tensor.shape[0]):
        label = label_tensor[x]
        predict = predict_tensor[x]
        predict = predict[predict != -1].numpy().tolist()
        label = label[label != -1].numpy().tolist()
        predict_list.append(predict)
        label_list.append(label)
    return f1_score(label_list, predict_list, tag, lable_type)


if __name__ == "__main__":
    # BIO
    tar_path = [[0, 1, 0, 0, 2, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 2, 2, 1, 0]]
    pre_path = [[0, 1, 0, 0, 2, 1, 0, 1, 2, 0, 0, 1, 2], [0, 1, 2, 2, 1, 0]]
    tag = 'entity'
    print(f1_score(tar_path, pre_path, tag, "BIO"))

    # BIEO
    tar_path = [[3, 0, 3, 2, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3], [0, 0, 1, 0, 2, 0, 0, 2, 3, 3, 1, 0]]
    pre_path = [[3, 0, 3, 2, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3], [0, 0, 1, 0, 2, 0, 0, 2, 3, 3, 1, 0]]
    tag = 'entity'
    print(f1_score(tar_path, pre_path, tag, "BIEO"))

    # BIESO
    tar_path = [[0, 1, 3, 1, 4, 2, 3, 0, 4, 4, 0, 1, 2, 0, 4, 4], [1, 0, 1, 3, 2, 0, 1, 2, 3, 2, 4, 4]]
    pre_path = [[0, 1, 2, 2, 3, 0, 4, 4, 0, 1, 2, 0, 4, 4, 4, 4], [1, 0, 1, 3, 2, 0, 1, 2, 3, 4, 4, 4]]
    tag = 'entity'
    print(f1_score(tar_path, pre_path, tag, "BIESO"))

    # BIESO
    tar_path = [[4, 0, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4]]
    pre_path = [[4, 0, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4]]
    tag = 'entity'
    print(f1_score(tar_path, pre_path, tag, "BIESO"))

    tar_path = torch.LongTensor(
        [[0, 1, 3, 1, 4, 2, 3, 0, 4, 4, 0, 1, 2, 0, 4, 4, -1, -1],
         [1, 0, 1, 3, 2, 0, 1, 2, 3, 2, 4, 4, -1, -1, -1, -1, -1, -1]])
    pre_path = torch.LongTensor(
        [[0, 1, 2, 2, 3, 0, 4, 4, 0, 1, 2, 0, 4, 4, 4, 4, -1, -1],
         [1, 0, 1, 3, 2, 0, 1, 2, 3, 4, 4, 4, -1, -1, -1, -1, -1, -1]])

    print(score_predict(tar_path, pre_path))
