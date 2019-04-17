from net.bert_ner import Bert_CRF
import torch

import args
from torch.utils.data import DataLoader, SequentialSampler

from util.model_util import load_model

from util.score import get_tags_BIESO
from data_loader import init_params
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from data_processor import convert_text_to_ids
from mydataset import MyTensorDataset


class entity_extractor:
    def __init__(self):
        print('[INFO]加载分词器')
        self.processor, self.bertTokenizer = init_params()
        label_list = self.processor.get_labels()
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.tokenizer = BasicTokenizer()
        print('[INFO]分词器加载完毕')
        print('[INFO]加载模型')
        self.model = Bert_CRF()
        self.model.load_state_dict(load_model(args.output_dir))
        self.device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print('[INFO]模型加载完毕')

    def extract(self, text_list):
        input_list = []
        en_set = set()
        for text in text_list:
            text = list(text)
            if len(text) > args.max_seq_length - 2:
                text = text[:(args.max_seq_length - 2)]
            tokens = ["[CLS]"] + text + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = convert_text_to_ids(tokens, self.bertTokenizer)
            input_mask = [1] * len(input_ids)
            padding = [0] * (args.max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length
            ## output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
            ## 此外，也是为了适应crf
            output_mask = [1 for t in text]
            output_mask = [0] + output_mask + [0]
            output_mask += padding
            text = ''.join(text)
            input_list.append((text, input_ids, input_mask, segment_ids, output_mask))
        all_input_text = [inp[0] for inp in input_list]
        all_input_ids = torch.tensor([inp[1] for inp in input_list], dtype=torch.long)
        all_input_mask = torch.tensor([inp[2] for inp in input_list], dtype=torch.long)
        all_segment_ids = torch.tensor([inp[3] for inp in input_list], dtype=torch.long)
        all_output_mask = torch.tensor([inp[4] for inp in input_list], dtype=torch.long)
        data = MyTensorDataset(all_input_text, all_input_ids, all_input_mask, all_segment_ids, all_output_mask,
                               all_output_mask)
        sampler = SequentialSampler(data)
        iterator = DataLoader(data, sampler=sampler, batch_size=args.inference_batch_size)

        with torch.no_grad():
            for step, batch_data in enumerate(iterator):
                text_list, batch = batch_data
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, _, output_mask = batch
                bert_encode = self.model(input_ids, segment_ids, input_mask).cpu()
                predicts = self.model.predict(bert_encode, output_mask)
                for ix, predict in enumerate(predicts):
                    predict = predict[predict != -1]
                    pre_tags = get_tags_BIESO(predict.cpu().numpy().tolist())
                    pre_entities = [text_list[ix][tag[0]:tag[1] + 1] for tag in pre_tags]
                    en_set = en_set | set(pre_entities)
        return set(en_set)


if __name__ == '__main__':
    ee = entity_extractor()
    text_list = ['放眼未来，笔者相信华为，甚至更多的中国企业将持续创新发力，为全球市场带来更多卓越的产品', '有人肯定要问，一家便利店，卖卖日用品，要这么大面积干什么',
                 '尽管建立cepc的成本很高，但从外部专家的角度看，这将是高能物理学的重大变化之一',
                 '鲅鱼喜欢清凉的水，所以多半是生活在北方，像南方那这种温暖的水，鲅鱼是生活不了的，如果硬要生活的话，那可能就是一条死鱼，所以很多人能够吃到鲅鱼还是要到北方去的，因为只有在北方才能吃到鲅鱼，其他地方要想吃到鲅鱼都是不可能的',
                 '此次ai合成主播的升级版和新品在“两会”上的应用，引发全球媒体圈的关注，并对其进行了大篇幅报道'] * 64
    print(ee.extract(text_list))
