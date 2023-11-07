# Models

Due to the size of models, we store the model files separately from GitHub.

## How to download the pre-trained model (VulBERTa)

1. Download the compressed file from (https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p)  
	
3. Unzip the compressed file in this directory (e.g. models)  
`unzip pretraining_model.zip`

4. This will extract 1 folder: `VulBERTa`
5. Now, you can use this pre-trained model to fine-tune on a specific vulnerability detection dataset.


## How to download the fine-tuned models (VulBERTa-MLP, VulBERTa-CNN)

1. Download the compressed file from (https://1drv.ms/u/s!AueKnGqzBuIVkq4DAleeVbhSzuB87w?e=jdI83b)  

2. Unzip the compressed file in this directory (e.g. models)  
`unzip finetuning_models.zip`

3. This will extract multiple folders:

* `VB-MLP_{dataset-name}` (for all 6 datasets)
* `VB-CNN_{dataset-name}` (for all 6 datasets)

4. In total, there will be 12 different folders extracted from the compressed file.
5. Now, you can use these fine-tuned models to test/evaluate on a specific vulnerability detection dataset or even any C/C++ source code.

* However, performance of fine-tuned models varies *
