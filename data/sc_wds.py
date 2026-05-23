import webdataset as wds
import os
from glob import glob
from tqdm import tqdm

class SCWDS:
    def __init__(self):
        pass
    
    @classmethod
    def get_paths(cls, path, split):
        pass

    @classmethod
    def get_dataset(cls, path, split="train", shardshuffle=True):
        shards, length = cls.get_paths(path, split)
        return wds.WebDataset(shards, shardshuffle=shardshuffle).to_tuple("img").with_length(length)


class Imagenet_1K(SCWDS):
    SPLIT_LENGTHS = {
        "train": 1281167,
        "test": 100000,
        "val": 50000
    }

    @classmethod
    def get_paths(cls, path, split="train"):
        return glob(os.path.join(path, f"*-{split}-*.tar")), cls.SPLIT_LENGTHS[split]

    @classmethod
    def get_from_hf(cls, split="train", shardshuffle=1):
        if split == "test":
            shard = "https://huggingface.co/datasets/Stoned-Code/imagenet-1k_wds/resolve/main/data/imagenet-1k-test-{000000..000999}.tar"
        elif split == "val":
            shard = "https://huggingface.co/datasets/Stoned-Code/imagenet-1k_wds/resolve/main/data/imagenet-1k-val-{000000..000499}.tar"
        elif split == "train":
            shard = "https://huggingface.co/datasets/Stoned-Code/imagenet-1k_wds/resolve/main/data/imagenet-1k-train-{000000..000640}.tar"
        
        return wds.WebDataset(shard, shardshuffle=shardshuffle).to_tuple("img").with_length(cls.SPLIT_LENGTHS[split])
