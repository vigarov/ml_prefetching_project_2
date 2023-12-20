# CS-433: Machine Learning - Project 2: Transformer Models for Page Pre-fetching

In this project created for EPFL's CS-433: Machine Learning, we explore the use of a transformer model for page-prefetching. 

## Installation

You can find the requirements in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

## Project structure

The project is structured as follows:

```
data/
├── prepro/
├── processed/
└── raw/
dataset_gathering
models/
├── config.py
├── train.py  
├── infer.py
├── model.py
├── data_parser.py
├── dataset.py
├── make_tokens.py
├── runs/
└── trainings/
```


* In `data` you can find our different raw data. 
The raw data directly collected can be found in `raw`, the preprocessed data in `prepro` and the processed, fully-cleaned data which we use in training in `processed`.
* `dataset_gathering` contains the code used to collect the raw data.
* `models` contains the code used to train, evaluate and the actual code of the transformer model.
  * `config.py` which can be used to tweak its parameters, e.g. the number of layers, the number of heads, the number of epochs, etc.
  * `train.py` contains the code to train the model
  * `infer.py` contains the code to use the model for inference
  * `model.py` contains the code of the model itself
  * `data_parser.py` contains the code to parse the data
  * `dataset.py` contains the code to create the dataset, parsing the raw data in a structure usable by our model and tokenizing it
  * `make_tokens.py` contains the tokenizer code, both for the input and the output
  * `runs` contains the tensorboard logs, which you can use to visualize the training
  * `trainings` contains the saved models, which you can use for inference and for further training

## Usage

### Training

To train the model on our dataset, simply run the `train.py` script, i.e.:
```bash
python train.py
```

You can tweak the parameters of the model in the `config.py` file.

### Inference

To use the model for inference, simply run the `infer.py` script, i.e.:
```bash
python infer.py
```

You can define your input string in the `infer.py` file (`data` parameter) and the maximum length of the output (`max_length` parameter).

## Authors

Victor Garvalov @vigarov, Alex Müller @ktotam1, Thibault Czarniak @t1b00.

## Acknowledgments

Thank you to Professors Martin Jaggi, Nicolas Flammarion, Sasha, and our wonderful TAs. 




