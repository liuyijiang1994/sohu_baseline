import json
import torch
import args

data = torch.load(args.entity_torch_data)
train_data = data['valid']
for i, (seq, label) in enumerate(zip(train_data['seq'], train_data['label'])):
    print(seq)
