# Autoencoder Trainer
A repository I created to train an autoencoder. The autoencoder takes an image and compresses it into latent space through a bottleneck. The compressed data can then be parsed through the decoder to be reconstructed. Things I'm working on here are going to help me in a neural codec project in the near future.

## Dataset
The dataset was processed into unstructured [WebDataset](https://github.com/webdataset/webdataset) format without any labels.
- [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

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
