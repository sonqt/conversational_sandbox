import argparse
import os
import json
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


def parseargs(arglist):
    """ This is my version of an argument parser.
        Parameters:
            arglist: the command line list of args
        Returns:
            the result of parsing with the system parser
    """
    parser = argparse.ArgumentParser()

    for onearg in arglist:
        if len(onearg) == 5:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3], default=onearg[4])
        else:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3])

    args = parser.parse_args()

    return args
def processDialog(args, dialog):
    """
    Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating.
    Returns the processed dialog entry where text has been replaced with a list of
    tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
    """
    label_metadata = "conversation_has_personal_attack" if args.corpus_name == "wikiconv" else "has_removed_comment"
    # Name of the utterance metadata field that contains comment-level toxicity labels, if any. Note
    # that CRAFT does not strictly need such labels, but some datasets like the wikiconv data do include
    # it. For custom datasets it is fine to leave this as None.
    utt_label_metadata = "comment_has_personal_attack" if args.corpus_name == "wikiconv" else None
    processed = []
    for utterance in dialog.iter_utterances():
        # skip the section header, which does not contain conversational content
        if args.corpus_name == 'wikiconv' and utterance.meta['is_section_header']:
            continue
        processed.append({"text": utterance.text, "is_attack": int(utterance.meta[utt_label_metadata]) if utt_label_metadata is not None else 0, "id": utterance.id})
    if utt_label_metadata is None:
        # if the dataset does not come with utterance-level labels, we assume that (as in the case of CMV)
        # the only labels are conversation-level and that the actual toxic comment was not included in the
        # data. In that case, we must add a dummy comment containing no actual text, to get CRAFT to run on 
        # the context preceding the dummy (that is, the full prefix before the removed comment)
        processed.append({"text": "", "is_attack": int(dialog.meta[label_metadata]), "id": processed[-1]["id"] + "_dummyreply"})
    return processed

def corpus2dataset(args, corpus, split=None, last_only=False, shuffle=False):
    """
    Load context-reply pairs from the Corpus, optionally filtering to only conversations
    from the specified split (train, val, or test).
    Each conversation, which has N comments (not including the section header) will
    get converted into N-1 comment-reply pairs, one pair for each reply 
    (the first comment does not reply to anything).
    Each comment-reply pair is a tuple consisting of the conversational context
    (that is, all comments prior to the reply), the reply itself, the label (that
    is, whether the reply contained a derailment event), and the comment ID of the
    last comment in the context (for later use in re-joining with the ConvoKit corpus).
    The function returns a list of such pairs.
    """
    dataset_dict = {
        "context": [],
        "id": [],
        "labels": []
    }
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        if split is None or convo.meta['split'] == split:
            dialog = processDialog(args, convo)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog)-1]
            for idx in iter_range:
                label = dialog[idx]["is_attack"]
                # when re-joining with the corpus we want to store forecasts in
                # the last comment of each context (i.e. the comment directly
                # preceding the reply), so we must save that comment ID.
                comment_id = dialog[idx-1]["id"]
                # gather as context all utterances preceding the reply
                context = [u["text"] for u in dialog[:idx]]
                dataset_dict["context"].append(context)
                dataset_dict["id"].append(comment_id)
                dataset_dict["labels"].append(label)
    if shuffle:
        return Dataset.from_dict(dataset_dict).shuffle(seed=2024)
    else:
        return Dataset.from_dict(dataset_dict)

def find_max_len(tokens_list):
    index = 0
    for i, tokens in enumerate(tokens_list):
        if len(tokens) > len(tokens_list[index]):
            index = i
    return index
def tokenize_context(context, tokenizer, max_len=512):
    tokenized_context = []
    for utterance in context:
        tokenized_context.append(tokenizer.encode(utterance, add_special_tokens=False))
    def truncate_context(tokenized_context, max_len):
        if sum([len(utterance) for utterance in tokenized_context])\
           + len(tokenized_context) + 1 < max_len:
            final_context = [tokenizer.cls_token_id]
            for utterance in tokenized_context:
                final_context += utterance + [tokenizer.sep_token_id]
            
            padding = [tokenizer.pad_token_id] * (max_len - len(final_context))
            mask = [1 for _ in range(len(final_context))] + [0 for _ in range(len(padding))]
            input_ids = torch.tensor(final_context + padding)
            mask = torch.tensor(mask)
            return {"input_ids": input_ids, "attention_mask": mask}
        
        while sum([len(utterance) for utterance in tokenized_context])\
              + len(tokenized_context) + 1 > max_len:
            truncate_idx = find_max_len(tokenized_context)
            tokenized_context[truncate_idx] = tokenized_context[truncate_idx][1:]
        final_context = [tokenizer.cls_token_id]
        for utterance in tokenized_context:
            final_context += utterance + [tokenizer.sep_token_id]
            input_ids = torch.tensor(final_context)
            mask = torch.tensor([1 for _ in range(len(final_context))])
            
        assert input_ids.shape[0] == max_len

        return {"input_ids": input_ids, "attention_mask": mask}
    return truncate_context(tokenized_context, max_len)

@torch.inference_mode
@torch.no_grad
def evaluateDataset(dataset, finetuned_model, device, threshold=0.5, temperature=1.0):
    finetuned_model = finetuned_model.to(device)
    convo_ids = []
    preds = []
    scores = []
    for data in tqdm(dataset):
        input_ids = data['input_ids'].to(device, dtype = torch.long).reshape([1,-1])
        attention_mask = data['attention_mask'].to(device, dtype = torch.long).reshape([1,-1])
        outputs = finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits / temperature, dim=-1)
        convo_ids.append(data["id"])
        raw_score = probs[0,1].item()
        preds.append(int(raw_score > threshold))
        scores.append(raw_score)
    return pd.DataFrame({"prediction": preds, "score": scores}, index=convo_ids)
@torch.inference_mode
@torch.no_grad
def batchevaluateDataset(dataset, finetuned_model, device, batch_size=64, threshold=0.5, temperature=1.0):
    finetuned_model = finetuned_model.to(device)
    convo_ids = []
    preds = []
    scores = []
    
    input_ids = []
    attention_mask = []
    ids = []
    for data in tqdm(dataset):
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        convo_ids.append(data["id"])
        # print(data['input_ids'].shape)
        if len(input_ids) == batch_size:
            input_ids = torch.stack(input_ids, dim=0).to(device, dtype = torch.long)
            attention_mask = torch.stack(attention_mask).to(device, dtype = torch.long)
            outputs = finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits / temperature, dim=-1)
            raw_score = probs[:,1].tolist()
            scores.extend(raw_score)
            
            input_ids = []
            attention_mask = []
    if len(input_ids) != 0:
        input_ids = torch.stack(input_ids, dim=0).to(device, dtype = torch.long)
        attention_mask = torch.stack(attention_mask).to(device, dtype = torch.long)
        outputs = finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits / temperature, dim=-1)
        raw_score = probs[:,1].tolist()
        scores.extend(raw_score)
        
    return pd.DataFrame({"score": scores}, index=convo_ids)
def acc_with_threshold(y_true, y_score, thresh):
    y_pred = (y_score > thresh).astype(int)
    return (y_pred == y_true).mean()

def full_evaluate(args, corpus, tokenized_dataset):
    """
    INPUT:
        saved_model_path: models are saved after each epoch.
        tokenized_dataset:
            tokenized_dataset['val_for_tuning']
            tokenized_dataset['test']
    """
    label_metadata = "conversation_has_personal_attack" if args.corpus_name == "wikiconv" else "has_removed_comment"
    utt_label_metadata = "comment_has_personal_attack" if args.corpus_name == "wikiconv" else None
    
    # Loop through all saved models to find the best model on val_for_tuning
    # Evaluate the best model on test
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    checkpoints = os.listdir(args.output_dir)
    best_val_accuracy = 0
    for cp in checkpoints:
        full_model_path = os.path.join(args.output_dir, cp)
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        val_scores = evaluateDataset(tokenized_dataset["val_for_tuning"], finetuned_model, "cuda")
        # for each CONVERSATION, whether or not it triggers will be effectively determined by what the highest score it ever got was
        highest_convo_scores = {c.id: -1 for c in corpus.iter_conversations(lambda convo: convo.meta['split']=="val")}
        for utt_id in val_scores.index:
            parent_convo = corpus.get_utterance(utt_id).get_conversation()
            utt_score = val_scores.loc[utt_id].score
            if utt_score > highest_convo_scores[parent_convo.id]:
                highest_convo_scores[parent_convo.id] = utt_score
        val_convo_ids = [c.id for c in corpus.iter_conversations(lambda convo: convo.meta['split'] == 'val')]
        val_labels = np.asarray([int(corpus.get_conversation(c).meta[label_metadata]) for c in val_convo_ids])
        val_scores = np.asarray([highest_convo_scores[c] for c in val_convo_ids])
        
        # use scikit learn to find candidate threshold cutoffs
        _, _, thresholds = roc_curve(val_labels, val_scores)
        accs = [acc_with_threshold(val_labels, val_scores, t) for t in thresholds]
        
        best_acc_idx = np.argmax(accs)
        if accs[best_acc_idx] > best_val_accuracy:
            best_val_accuracy = accs[best_acc_idx]
            best_threshold = thresholds[best_acc_idx]
    
    forecasts_df = evaluateDataset(tokenized_dataset["test"], finetuned_model, "cuda", threshold=best_threshold)
    # We will add a metadata entry to each test-set utterance signifying whether, at the time
    # that CRAFT saw the context *up to and including* that utterance, CRAFT forecasted the
    # conversation would derail. Note that in datasets where the actual toxic comment is
    # included (such as wikiconv), we explicitly do not show that comment to CRAFT (since
    # that would be cheating!), so that comment will not have an associated forecast.
    for convo in corpus.iter_conversations():
        # only consider test set conversations (we did not make predictions for the other ones)
        if convo.meta['split'] == "test":
            for utt in convo.iter_utterances():
                if utt.id in forecasts_df.index:
                    utt.meta['forecast_score'] = forecasts_df.loc[utt.id].score
    
    conversational_forecasts_df = {
        "convo_id": [],
        "label": [],
        "score": [],
        "prediction": []
    }

    for convo in corpus.iter_conversations():
        if convo.meta['split'] == "test":
            conversational_forecasts_df['convo_id'].append(convo.id)
            conversational_forecasts_df['label'].append(int(convo.meta[label_metadata]))
            forecast_scores = [utt.meta['forecast_score'] for utt in convo.iter_utterances() if 'forecast_score' in utt.meta]
            conversational_forecasts_df['score'] = np.max(forecast_scores)
            conversational_forecasts_df['prediction'].append(int(np.max(forecast_scores) > best_threshold))

    conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df).set_index("convo_id")
    test_labels = conversational_forecasts_df.label
    test_preds = conversational_forecasts_df.prediction
    test_acc = (test_labels == test_preds).mean()
    

    tp = ((test_labels==1)&(test_preds==1)).sum()
    fp = ((test_labels==0)&(test_preds==1)).sum()
    tn = ((test_labels==0)&(test_preds==0)).sum()
    fn = ((test_labels==1)&(test_preds==0)).sum()
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    test_fpr = fp / (fp + tn)
    test_f1 = 2 / (((tp + fp) / tp) + ((tp + fn) / tp))

    result_dict = {'accuracy': test_acc, 
                    'f1': test_f1, 
                    'precision': test_precision,
                    'recall': test_recall,
                    'false positive rate': test_fpr}
    result_file = os.path.join(args.output_dir, "result.json")
    with open(result_file, 'w') as f:
        json.dump(result_dict, f)
    return