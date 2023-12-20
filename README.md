# Data Pruning
This repository is for a data pruning project based on ["Beyond neural scaling laws: beating power law scaling via data pruning"](https://arxiv.org/abs/2206.14486). In this repository, we have provided a simple but effective tool to find and separate hard and easy samples in a data corpus. By doing so, we can easily select samples that are suitable for our tasks. 

## Description
To use this tool consider two main directories for pruning: 1. corpus - 2. target. The corpus directory consists of images that we want to select for training. A target directory is a directory that we prune data based on it. For instance, assume we have 100K images and we want to select the 1% percentile of the hardest images. To define the hardness metric, we need to have a target dataset. This dataset has about 500 images per class. So, we can use our tool to find the hardest images based on our defined dataset.

We have two main modes for data pruning: 1. supervised, and 2. unsupervised. In supervised mode, the structure of the corpus dataset and target dataset is the same. The structure of data in this mode is as below:

```
# corpus_dataset/
#   cls_1/
#      img_1.jpg
#      img_2.jpg
#      ...
#   cls_2/
#   ...

# target_dataset/
#   cls_1/
#      img_1.jpg
#      img_2.jpg
#      ...
#   cls_2/
#   ...
```

In other words, the corpus dataset's classes are the same as the target dataset. 

In unsupervised mode, the corpus dataset is a mix of images from different classes all in a folder. The structure of data in this mode is as below:

```
# corpus_dataset/
#   dummy_folder/
#      img_1.jpg
#      img_2.jpg
#      ...

# target_dataset/
#   cls_1/
#      img_1.jpg
#      img_2.jpg
#      ...
#   cls_2/
#   ...
```

In this tool, we can define a quantile for selecting samples. In default, the quantile is 0.9. It means that only 90% of images will be ignored and the remaining 10% of images will be selected. Selected images are the 10% hardest images in the corpus. 

By setting --num-clusters with the desired number, you can control the procedure of data pruning. The number of classes in the target dataset is a good choice for this parameter. 

There is a flag known as --difficulty, which can be 'easy' or 'hard'. Furthermore, you can see histograms of the difficulty distribution of images by using the --draw-histogram flag. 

## Method
The main idea behind data pruning is finding samples that are slightly out of distribution. In other words, in the first stage, the k-means algorithm finds clusters of samples based on the --num-clusters value. Then, with a defined metric the hardest and easiest samples in a defined quantile can be discovered. This metric can be the difference between a sample and the corresponding centroid. 

## Usage
To use DataPruning code, follow these steps:

1. Create directories for corpus and target datasets, as mentioned above.

2. Install provided requirements.

3. Run the below code to do data pruning.
```
python main.py --data-path=[path_to_target_dataset] --corpus-path=[path_to_corpus_dataset] --quantile=[quantile] --mode=[mode: supervised or unsupervised] --difficulty=[difficulty: hard or easy] --num-clusters[value_of_k] --model-name=[name_of_model] --model-path=[path_to_checkpoint]  
```

4. Outputs will be in "easy samples" or "hard samples" directories, based on the difficulty.



## Requirements
The following packages and libraries are required to run the stock price prediction model:
- Python 3.8 or higher
- Pytorch 1.13
- timm 0.6.7

Install the required dependencies by running `pip install -r requirements.txt`

---
## TODO
- [ ] ToDo.

---
## License

---
