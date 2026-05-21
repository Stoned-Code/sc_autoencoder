import webdataset as wds
import os
from glob import glob

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


DATASETS = {
    "imagenet-1k": {
        "object":Imagenet_1K,
        "splits": [
            "train",
            "test",
            "val"
        ]
    }
}

class SCWebDatasets:
    def __init__(self, root):
        self.root = root
    
    @staticmethod
    def get_data_length(path, split):
        if isinstance(path, list):
            if split is not None:
                shards = glob(os.path.join(path, f"*-{split}-*.tar"))
            else:
                shards = glob(os.path.join(path, "*.tar"))
            dataset = wds.WebDataset(shards)
            length = 0

            for _ in tqdm.tqdm(dataset, desc="Getting Dataset Length"):
                length += 1

            return length
        else:
            length = 0
            for _ in tqdm.tqdm(path, desc="Getting Dataset Length"):
                length += 1

            return length

    def create_dataset_from_shards(cls, shards, length, shardshuffle=True):
        dataset = wds.WebDataset(shards, shardshuffle=shardshuffle)

        return dataset.to_tuple("img").with_length(length)

    def get_datasets(self, datasets: list, split="train", shardshuffle=True):

        shards = []
        lengths = []
        
        for dataset in datasets:
            if dataset not in DATASETS:
                raise Exception(f"Dataset {dataset} does not exist...")

            dataset_obj = DATASETS[dataset]

            if split not in dataset_obj["splits"]:
                raise Exception(f"{dataset} doesn't have a {split} split...")
            
            data_shards, data_length = dataset_obj["object"].get_paths(os.path.join(self.root, dataset, "data"), split)
            shards.extend(data_shards)
            lengths.append(data_length)
        
        return self.create_dataset_from_shards(shards, sum(lengths), shardshuffle)
