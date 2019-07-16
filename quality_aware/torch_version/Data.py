from torch.utils.data import Dataset
from torch import Tensor
import torch

def collate_fn(batch):
    batch_size=len(batch)
    visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label = zip(*batch)
    visual_tensor_1=torch.stack(visual_tensor_1)
    text_tensor_1=torch.stack(text_tensor_1)
    visual_tensor_2 = torch.stack(visual_tensor_2)
    text_tensor_2=torch.stack(text_tensor_2)
    rating_2=torch.stack(rating_2).reshape((batch_size,1))
    label=torch.stack(label)

    return visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label

class ExtDataset(Dataset):
    def __init__(self,data_list,text,visual,rating):
        self.data_list=data_list
        self.text=text
        self.visual=visual
        self.rating=rating
        self.len=len(data_list)

    def __getitem__(self, index):
        tup=self.data_list[index]
        item_key=tup[0]
        target_key=tup[1]
        label=tup[2]

        visual_tensor_1=torch.from_numpy(self.visual[item_key].reshape((-1))).float()
        text_tensor_1=torch.from_numpy(self.text[item_key].reshape((-1))).float()
        visual_tensor_2 = torch.from_numpy(self.visual[target_key].reshape((-1))).float()
        text_tensor_2 = torch.from_numpy(self.text[target_key].reshape((-1))).float()
        rating_2=torch.tensor(self.rating[target_key]).float()
        label = torch.tensor(tup[2]).byte()

        return visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label

    def __len__(self):
        return self.len

