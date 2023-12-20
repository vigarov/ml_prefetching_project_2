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
### Configuration

In `config.py' you can tweak many of the model's parameters, such as the number of layers, the number of heads, the number of epochs, etc, but also parameters for the tokenizers.
Here, we explain how these parameters will affect the model and what values they can take
##### Global parameters
* `DATA_PATH`: the folder where the data file is
* `OBJDUMP_PATH`: folder where `objdump` output for libraries used by traced programs is
* `GENERATOR_PREFIX`: prefix of the folder where the generator will be saved
* `SEED_FN`: the seed to use for the generator 
* `STATE_FN`: the name of the state to use for the generator state saving / loading
* `TRACE_TYPE`: the type of trace to use, either `fltrace` or `bpftrace`
#### Features used
`BPF_FEATURES` is a list of features used to train / infer with the model, collected with the BPF method. It contains:
* `prev_faults` which is an hex-address list, containing the addresses of the previous page faults
* `flags` which is a bitmap, containing the flags of the page fault, includes `rW`.
* `ip` which is the instruction pointer of the CPU
* `regs` which is a list of the values of the registers of the CPU

`FL_FEATURES` is a list of features used to train / infer with the model, collected with the `fltrace`. It contains:
* `prev_faults` which is an hex-address list, containing the addresses of the previous page faults
* `rW`: whether the page was read from or written to
* `ips`: stack trace of the program

`OUTPUT_FEATURES` contains the output features of the model, which is by default only one element: the hex-addresses of the next pages to prefetch.

#### [Meta]Transformer parameters
Transformer parameters are set in the the `TransformerModelParams` class and Meta Transformer parameters are set in `MetaTransformerParams`, both in `config.py`. They are:
* `d_model`: the dimension of the model
* `T`: the number of transformer block layers
* `H`: the numbe of attention heads per transformer layer
* `dropout`: the dropout rate
* `d_ff`: the dimension of the feed-forward layer

#### The configuration itself
In the `get_config` function in `config.py`, we create the configuration, you can modify most parameters there. 
* `bpe_special_tokens`: the special tokens used by the tokenizers. Default: `[UNK]`
* `pad_token`: the padding token used by the tokenizers. Default: `[[PAD]]`
* `list_elem_separation_token`: the token used to separate elements in a list. Default: ` ` (space) _For use, see comment in `TokenizerWrapper` of `special_tokenizers.py`_
* `feature_separation_token`: the token used to separate features in a list. Default: `[FSP]`
*  `start_stop_generating_tokens`: the tokens used to indicate the start and the end of a sentence. Default: `[GTR]` and `[GTP]`
* `batch_size`: the batch size to use for training. Default: 16
* `num_epochs`: the number of epochs to train for. Default: 10
* `lr`: the learning rate to use for training. Default 10^(-4)
* `trace_type`: the type of trace to use, see `TRACE_TYPE` above. Default: `fltrace`, choose with `bpftrace`
* `train_on_trace`: for `fltrace` only, whether we train on one trace or multiple
* `datasource`: name of the benchmark on which we train
* `subsample`: the subsampling rate to use for the data. Default: 0.2
* `objdump_path`: see `OBJDUMP_PATH` above
* `model_folder`: the folder where the model will be saved. Default: `models`
* `preload`: which version of the model to preload, default `latest` (takes the highest epoch number)
* `tokenizer_files`: format string path to the the tokenizer files. Default: `trained_tokenizers/[TRACETYPE]/tokenizer_[src,tgt].json`
* `train_test_split`: the train / test split to use. Default: 0.75
* `attention_model`: the type of model to use for attention. Default: `transformer`, choose with `retnet`
* `attention_model_params`: the parameters of the attention model. Default: `TransformerModelParams`, not needed for RetNet
* `decode_algorithm`: the algorithm to use for decoding. Default: `beam` (beam search), choose with `greedy` (greedy decode)
* `beam_size`: if `decode_algorithm` is `beam`, the beam size to use. Default: 4
* `past_window`: the size of the past window of previous faults to use. Default: 10
* `k_predictions`: the number of predictions to make. Default: 10
* `code_window`: tuple of number of instructions before and after the instruction pointer, i.e. code window around IP. Default (1,2)
* `input_features`: the features to use as input. Default: `BPF_FEATURES`, choose with `FL_FEATURES`
* `output_features`: the features to use as output. Default: `OUTPUT_FEATURES`
* `base_tokenizer`: the base tokenizer to use. Default: `hextet`, choose with `bpe`, `text`. See tokenizers section.
* `embedding_technique`: the embedding technique used on the tokens. See embeddings. Default: `tok_concat`, choose with `onetext`, `meta_transformer` and `embed_concat`
* `meta_transformer_params`: the parameters of the meta transformer. Default: `MetaTransformerParams`
* `page_masked`: for `bpftrace` only, convert map all accesses to a page. This is default behavior with `fltrace` #TODO VICTOR
* `max_weight_save_history`: used when `mass_train == True` in training. Defines how many epochs we should save at most. Default: 3

### Tokenizers

In `models/trained_tokenizers/special_tokenizers.py`, we define generic classes of tokenizers, which are then trained on a specific vocabulary.
We have three generic classes:
* `SimpleCustomVocabTokenizer`
* `TokenizerWrapper`
* `ConcatTokenizer`
Details can be found in the docstrings of the classes.
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




