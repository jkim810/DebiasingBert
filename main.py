#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:25:27 2019

@author: junbum
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:29:03 2019

@author: Junbum
"""

import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt


# import logging
# logging.basicConfig(level=logging.INFO)

# Generalized Bert for Language Models
class Bert(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()
       
    def __call__(self, text, k = 10, verbose = False):
        # append [CLS], [SEP] to data and tokenize
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
       
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        self.model.to('cuda')
        self.male_pronoun = ["he", "him", "his", "himself"]
        self.female_pronoun = ["she", "her", "her", "herself"]
   
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
       
        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 4
       
        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):
            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()
           
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

             
            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to("cuda") for t in batch)
                inputs, labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss = self.model(inputs, masked_lm_labels=labels)
                # print(loss)
                # loss = output[0]
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
         
if __name__ == "__main__":
   
    bert = Bert()
    hbert = HardGNBert()
    sbert = SoftGNBert()
    sbert.finetune()
   
    text = 'I want to [MASK] the car because it is cheap .'
    bert(text)
   
    text4 = "He works as a [MASK] ."
    male = bert(text4,200)
    hard1 = hbert(text4, 10)
   
    text5 = "She works as a [MASK] ."
    female = bert(text5,200)
    
    
    np.random.seed(41)
    x = list(female_bias.values())
    y = np.random.random(size=len(female_bias))
    plt.scatter(x,y)
    annotation = list(female_bias.keys())
    bottomK = torch.topk(torch.tensor(x), k = 5, largest = False).indices
    for i, txt in enumerate(annotation):
        if i in bottomK:
            plt.annotate(txt, (x[i],y[i]))
 
    # Extreme male words
    x = list(male_bias.values())
    y = np.random.random(size=len(male_bias))
    plt.scatter(x,y)
    annotation = list(male_bias.keys())
    topK = torch.topk(torch.tensor(x), k = 5).indices
    for i, txt in enumerate(annotation):
        if i in topK:
            plt.annotate(txt, (x[i],y[i]))
       
    plt.title("Quantification of Gender Bias in BERT")
    plt.xlabel("Gender Bias")
    plt.legend(["Inbetween","Female Exclusive Occupation","Male Exclusive Occupation"], loc=2)
    plt.show()
