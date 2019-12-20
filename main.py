# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:29:03 2019

@author: Junbum
"""

import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from pytorch_pretrained_bert import BertAdam
from tqdm import  trange
import numpy as np
import matplotlib.pyplot as plt

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Generalized Bert for Language Models
class Bert(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.male_pronoun = ["he", "him", "his", "himself"]
        self.female_pronoun = ["she", "her", "her", "herself"]

    def __call__(self, text, k = 10, verbose = False):
        # append [CLS], [SEP] to data and tokenize input
        tokenized_text = self.tokenize(text)
        
        # return error if [MASK] not in text
        try:
            masked_index = tokenized_text.index("[MASK]")
        except ValueError:
            print("[MASK] not found in text")
            return -1
        
        # fetch token indices for each token (one hot encoding) / put it into tensors
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        
        # get positional index of each words
        segments_ids = [0] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids])
        
        # make predictions for a given language model
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        
        # fetch top k likely words and their log likelihood for [MASK]
        topk = torch.topk(predictions[0, masked_index], k)
        topk_value = topk.values.tolist()
        topk_index = topk.indices.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(topk_index)
        
        d = dict()
        for token, likelihood in zip(tokens, topk_value):
            d[token] = likelihood
            
        if verbose:
            print("Sentence to predict:", text)
            print("")
            for token in d:
                print("{:<16}{:.4f}" .format(token,d[token]))
            
        return d
    
    def tokenize(self, text):
        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        return tokenized_text
    
# Hard Gender Neutralized Bert model
# This would take the return gender neutralized representation using pairs
class HardGNBert(Bert):
    def __init__(self):
        super().__init__()
    
    def __call__(self, text, k=10, verbose = False):
        tokenized_text = self.tokenize(text)
        
        try:
            masked_index = tokenized_text.index("[MASK]")
        except ValueError:
            print("[MASK] not found in text")
            return -1
        
        # Hard Debiasing through taking the mean of all conjugate paris
        textlist = [tokenized_text]
        for i, word in enumerate(tokenized_text):
            if word in self.male_pronoun:
                idx = self.male_pronoun.index(word)
                conjugate_sentence = tokenized_text[:i] + [self.female_pronoun[idx]] + tokenized_text[i+1:]
                textlist.append(conjugate_sentence)
            if word in self.female_pronoun:
                idx = self.female_pronoun.index(word)
                conjugate_sentence = tokenized_text[:i] + [self.male_pronoun[idx]] + tokenized_text[i+1:]
                textlist.append(conjugate_sentence)
        
        
        predictions = []
        for token_text in textlist:
            # fetch token indices for each token (one hot encoding) / put it into tensors
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(token_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            
            # get positional index of each words
            segments_ids = [0] * len(token_text)
            segments_tensors = torch.tensor([segments_ids])
            
            # make predictions for a given language model
            with torch.no_grad():
                predictions.append(self.model(tokens_tensor, segments_tensors))
        
        mean = torch.mean(torch.stack(predictions),0)
        
        # fetch top k likely words and their log likelihood for [MASK]
        topk = torch.topk(mean[0, masked_index], k)
        topk_value = topk.values.tolist()
        topk_index = topk.indices.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(topk_index)
        
        d = dict()
        for token, likelihood in zip(tokens, topk_value):
            d[token] = likelihood
            
        if verbose:
            print("Sentence to predict:", text)
            print("")
            for token in d:
                print("{:<16}{:.4f}" .format(token,d[token]))
            
        return d
    
# Soft Gender Neutralized Bert model
# This would take the return gender neutralized representation using pairs
class SoftGNBert(Bert):
    def __init__(self):
        super().__init__()
    
    # attention masking code from : https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # the official github repository for transformers
    def mask_tokens(self, inputs, tokenizer):
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens
    
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
    
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    # code reference for fine-tuning pytorch bert: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    # the tutorial code is for bert-for classification, and i specifically editted this example for BERT-Language models
    def finetune(self, train_file = "data/GNtrain_clean.txt", valid_file="data/GNvalid_clean.txt"):
        MAX_LEN = 64
        train = [line.rstrip('\n') for line in open(train_file)]
        valid = [line.rstrip('\n') for line in open(valid_file)]

        # tokenize text
        train_texts = [self.tokenize(sent) for sent in train]
        valid_texts = [self.tokenize(sent) for sent in valid]
        
        # get one hot encoding ids for texts
        train_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in train_texts]
        valid_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in valid_texts]

        # pad zeros to sequence for flexibility of sentence length
        train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        valid_ids = pad_sequences(valid_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        
        train_inputs = torch.tensor(train_ids)
        validation_inputs = torch.tensor(valid_ids)

        # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
        batch_size = 16
        
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        
        train_inputs, train_labels = self.mask_tokens(train_inputs, tokenizer)
        train_data = TensorDataset(train_inputs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        
        validation_inputs, validation_labels = self.mask_tokens(validation_inputs, tokenizer)
        validation_data = TensorDataset(validation_inputs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        
        self.model.cuda()
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        
        # This variable contains all of the hyperparemeter information our training loop needs
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=2e-5,
                             warmup=.1)
        
        train_loss_set = []
        epochs = 8
        
        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):
            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_steps = 0

              
            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to("cuda") for t in batch)
                inputs, labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss = self.model(inputs, masked_lm_labels=labels)
                train_loss_set.append(loss.item())
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                
                
                # Update tracking variables
                tr_loss += loss.item()
                #nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            self.model.eval()
            
            # Tracking variables 
            eval_loss = 0.0
            nb_eval_steps = 0
            
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to("cuda") for t in batch)
                inputs, labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    lm_loss = self.model(inputs, masked_lm_labels=labels)
                    #lm_loss = outputs[0]
                    eval_loss += lm_loss.mean().item()
                    nb_eval_steps += 1
            
            print("Validation Accuracy: {}".format(eval_loss/nb_eval_steps))
        return train_loss_set


def pretty(df, i = 20):
    for idx, key in enumerate(df):
        if idx == i:
            break
        print("{:<16}{:.4f}" .format(key,df[key]))

def plot_and_annotate(df, plt, k=5, mode="top"):
    x = list(df.values())
    y = np.random.random(size=len(df))
    plt.scatter(x,y)
    annotation = list(df.keys())
    topK, bottomK = None, None
    if mode == "top":
        topK = torch.topk(torch.tensor(x), k = k).indices
    elif mode == "bottom":
        bottomK = torch.topk(torch.tensor(x), k = k, largest=False).indices
    elif mode == "both":
        topK = torch.topk(torch.tensor(x), k = k).indices
        bottomK = torch.topk(torch.tensor(x), k = k, largest=False).indices
    else:
        return # unexpected intput
    for i, txt in enumerate(annotation):
        if torch.is_tensor(topK) and i in topK:
            plt.annotate(txt, (x[i],y[i]))
        if torch.is_tensor(bottomK) and i in bottomK:
            plt.annotate(txt, (x[i],y[i]))
    
if __name__ == "__main__":
    bert = Bert()
    hbert = HardGNBert()
    sbert = SoftGNBert()
    sbert.finetune()
    
    text4 = "He works as a [MASK] ."
    male = bert(text4, 200)
    hard1 = hbert(text4, 200)
    soft_male = sbert(text4, 200)
    
    text5 = "She works as a [MASK] ."
    female = bert(text5, 200)
    hard2 = hbert(text5, 200)
    soft_female = sbert(text5, 200)
    
    male_job = set(male.keys())
    female_job = set(female.keys())
    
    neutral_job = male_job.intersection(female_job)
    all_job = male_job.union(female_job)
    male_exclusive_job = all_job - female_job
    female_exclusive_job = all_job - male_job
    
    # test 1    
    bias = dict()
    for job in neutral_job:
        bias[job] = male[job] - female[job]
    
    female_bias = dict()
    for job in female_exclusive_job:
        female_bias[job] = -female[job]
    
    male_bias = dict()
    for job in male_exclusive_job:
        male_bias[job] = male[job]
    
    # test 2
    bias2 = 0
    bias_length = 0
    for job in all_job:
        text = "[MASK] is a " + job + " ."
        result = bert(text,10)
        if "he" in result and "she" in result:
            bias2 += result["he"] - result["she"]
            bias_length += 1
    print(bias2/bias_length)
    
    # What is supposedely gender neutral words
    np.random.seed(45)
    plt.figure(figsize=(11, 8), dpi=400)
    
    plot_and_annotate(bias,plt,mode="both")
    plot_and_annotate(female_bias,plt,mode="bottom")
    plot_and_annotate(male_bias,plt,mode="top")
    plt.title("Quantification of Gender Bias in BERT")
    plt.xlabel("Gender Bias")
    plt.legend(["Inbetween","Female Exclusive Occupation","Male Exclusive Occupation"])
    
