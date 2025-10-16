**Using a Convolutional Neural Network to predict stages of Alzheimer's / Dementia through MRI imagery**

This repo consists of modifications to the CIFAR-10 CNN / Datasets to support MRI scanning and classification. 
Additionally, consists of several experiments, example classifications, test accuracy & runtime. 

Download Operation:
* git clone git@github.com:abrizu/480-NN-Project-MRI.git
* python -m venv .venv
* .venv/scripts/activate
* pip freeze > requirements.txt

Data can be downloaded 2 ways:

External Download - 
Download the MRI Dataset from https://www.kaggle.com/datasets/ninadaithal/imagesoasis
Extract the dataset into the project root folder.

Auto Download (Using API token) -
Work In Progress

* 'Data' folder is excluded from project due to size.

To Execute:
py mri.py
