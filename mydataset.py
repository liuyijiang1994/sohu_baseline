from torch.utils.data import Dataset


class MyTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, text_list, all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask):
        tensors = [all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.text_list = text_list

    def __getitem__(self, index):
        return self.text_list[index], tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
