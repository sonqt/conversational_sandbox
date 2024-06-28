from utils import parseargs, corpus2dataset, loadtraindataset,\
                 tokenize_context, full_evaluate
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import unicodedata
import itertools
from convokit import download, Corpus
from tqdm import tqdm
from sklearn.metrics import roc_curve

from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import evaluate
import logging
import re

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR) # Avoid printing warning from Tokenizer

def main(args):
    if args.corpus_name == "wikiconv":
        corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
    elif args.corpus_name == "cmv":
        corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
    else:
        raise Exception("Sorry, no corpus_name matched the input {}.\
         Please input a valid corpus_name [wikiconv, cmv]".format(args.corpus_name))
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=512, truncation_side="left", padding_side="right"
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    dataset_dict = {
        "train": corpus2dataset(args, corpus, "train", last_only=True), 
        "val": corpus2dataset(args, corpus, "val", last_only=True),
        "val_for_tuning": corpus2dataset(args, corpus, "val"),
        "counterfactual_val": corpus2dataset(args, corpus, "val"),
        "test": corpus2dataset(args, corpus, "test"),
        "counterfactual_test": corpus2dataset(args, corpus, "test")
    }
    def map_data(dataset, tokenizer, counterfactual=False):
        """
        dataset: an instance of datasets.Dataset class
        """
        # new_columns = tokenize_context(dataset["context"], tokenizer, counterfactual=counterfactual)
        # for column in new_columns:
        #     dataset.add_column(name=column, column=new_columns[column])
        dataset = dataset.map(lambda inst: tokenize_context(inst["context"], tokenizer, counterfactual=counterfactual))
        dataset.remove_columns("context")
        return dataset
    dataset_dict["train"] = map_data(dataset_dict["train"], tokenizer)
    dataset_dict["val"] = map_data(dataset_dict["val"], tokenizer)
    dataset_dict["val_for_tuning"] = map_data(dataset_dict["val_for_tuning"], tokenizer)
    dataset_dict["counterfactual_val"] = map_data(dataset_dict["counterfactual_val"], tokenizer, counterfactual=True)
    dataset_dict["test"] = map_data(dataset_dict["test"], tokenizer)
    dataset_dict["counterfactual_test"] = map_data(dataset_dict["counterfactual_test"], tokenizer, counterfactual=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # load the corpus into PyTorch-formatted train, val, and test datasets
    tokenized_dataset = DatasetDict(dataset_dict)
    tokenized_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                ignore_mismatched_sizes=True,
                                                                num_labels=2)
    
    if args.do_train:
        def compute_metrics(eval_pred):
            cls_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return cls_metrics.compute(predictions=predictions, references=labels)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,  #6.7e-6 https://arxiv.org/pdf/2110.05111.pdf
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            prediction_loss_only=False,
            run_name=f"bertcraft_{args.corpus_name}",
            logging_steps=1,
            seed=args.random_seed,
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            compute_metrics=compute_metrics,
        )
        trainer.train()
    if args.do_eval:
        full_evaluate(args, corpus, tokenized_dataset)
    return 


args = parseargs([['-model', '--model_name_or_path', 'model_name', str],
                     ['-corpus', '--corpus_name', 'corpus_name', str],
                     ['-train', '--do_train', 'train_before_evaluate', bool],
                     ['-eval', '--do_eval', 'evaluate_or_not', bool],
                     ['-lr', '--learning_rate', 'learning_rate', float, 2e-5],
                     ['-bs', '--per_device_batch_size', 'number_of_samples_on_each_GPU', int, 8],
                     ['-epoch', '--num_train_epochs', 'num_train_epochs', int, 5],
                     ['-output', '--output_dir', 'output_directory', str],
                     ['-seed', '--random_seed', 'random_seed', int, 42]
                     ])
                     # type_context
print(f'ARGPARSE OPTIONS {args}')
main(args)
