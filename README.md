# Autoencoder Trainer
A repository I created to train an autoencoder.

## Datasets
- [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
- [Tagged Anime Illustrations](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations)

## Getting Started
Create conda environment
```
conda create -p .conda python=3.12
```
Install requirements.
```
python -m pip install -r requirements.txt
```

## Normalization
During training I started by normalizing the images by using the formula `(X - 127.5) / 127.5 (Formula A)` as well as the formula `X / 255.0 (Formula B)`. And after trying to train with both of them, I found that formula b works best.
