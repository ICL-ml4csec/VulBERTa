
# VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection

![VulBERTa architecture](VB.png)

## Overview
This paper presents VulBERTa, a deep learning approach to detect security vulnerabilities in source code. Our approach pre-trains a RoBERTa model with a custom tokenisation pipeline on real-world code from open-source C/C++ projects. The model learns a deep knowledge representation of the code syntax and semantics, which we leverage to train vulnerability detection classifiers. We evaluate our approach on binary and multi-class vulnerability detection tasks across several datasets (Vuldeepecker, Draper, REVEAL and muVuldeepecker) and benchmarks (CodeXGLUE and D2A). The evaluation results show that VulBERTa achieves state-of-the-art performance and outperforms existing approaches across different datasets, despite its conceptual simplicity, and limited cost in terms of size of training data and number of model parameters.

## Data
We provide all data required by VulBERTa.  
This includes:
 - Tokenizer training data
 - Pre-training data
 - Fine-tuning data

Please refer to the [data](https://github.com/ICL-ml4csec/VulBERTa/tree/main/data "data") directory for further instructions and details.

## Models
We provide all models pre-trained and fine-tuned by VulBERTa.  
This includes:
 - Trained tokenisers
 - Pre-trained VulBERTa model (core representation knowledge)
 - Fine-tuned VulBERTa-MLP and VulBERTa-CNN models

Please refer to the [models](https://github.com/ICL-ml4csec/VulBERTa/tree/main/models "models") directory for further instructions and details.

## Pre-requisites and requirements

In general, we used this version of packages when running the experiments:

 - Python 3.8.5
 - Pytorch 1.7.0
 - Transformers 4.4.1
 - Tokenizers 0.10.1

For an exhaustive list of all the packages, please refer to [requirements.txt](https://github.com/ICL-ml4csec/VulBERTa/blob/main/requirements.txt "requirements.txt") file.

## How to use

In our project, we uses Jupyterlab notebook to run experiments.  
Therefore, we separate each task into different notebook:

 - [Pretraining_VulBERTa.ipynb](https://github.com/ICL-ml4csec/VulBERTa/blob/main/Pretraining_VulBERTa.ipynb "Pretraining_VulBERTa.ipynb") - Pre-trains the core VulBERTa knowledge representation model using DrapGH dataset.
 - [Finetuning_VulBERTa-MLP.ipynb](https://github.com/ICL-ml4csec/VulBERTa/blob/main/Finetuning_VulBERTa-MLP.ipynb "Finetuning_VulBERTa-MLP.ipynb") - Fine-tunes the VulBERTa-MLP model on a specific vulnerability detection dataset.
 - [Evaluation_VulBERTa-MLP.ipynb](https://github.com/ICL-ml4csec/VulBERTa/blob/main/Evaluation_VulBERTa-MLP.ipynb "Evaluation_VulBERTa-MLP.ipynb") - Evaluates the fine-tuned VulBERTa-MLP models on testing set of a specific vulnerability detection dataset.
 - [Finetuning+evaluation_VulBERTa-CNN](https://github.com/ICL-ml4csec/VulBERTa/blob/main/Finetuning%2Bevaluation_VulBERTa-CNN.ipynb "Finetuning+evaluation_VulBERTa-CNN.ipynb") - Fine-tunes VulBERTa-CNN models and evaluates it on a testing set of a specific vulnerability detection dataset.

## Running VulBERTa-CNN or VulBERTa-MLP on arbitrary codes

Coming soon!

## Citation
Link to Arxiv: https://arxiv.org/abs/2205.12424  

Accepted as conference paper (oral presentation) at the International Joint Conference on Neural Networks (IJCNN) 2022  

```bibtex
@INPROCEEDINGS{hanif2022vulberta,

  author={Hanif, Hazim and Maffeis, Sergio},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)}, 
  title={VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-8}
  
}
```
