import torch as tt
import re
import numpy as np

from tqdm import tqdm as tqdm_notebook
from math import ceil
from copy import copy
from nltk import pos_tag

from utils import nltk_to_upos

class UnsupervisedBatch:
  def __init__(self, sents_batch):
    self.masked = sents_batch

class UnsupervisedBatchIterator:
  def __init__(self, sents, batch_size=8):
    self.sents = sents
    self.batch_size = batch_size
  
  def __iter__(self):
    self.start = 0
    return self
  
  def __next__(self):
    if self.start >= len(self.sents):
      raise StopIteration
    batch = self.sents[self.start:self.start+self.batch_size]
    batch = UnsupervisedBatch(batch)
    self.start += self.batch_size
    return batch
  
  def __len__(self):
    return ceil(len(self.sents)/self.batch_size)

def tokenize_batch(batch, tokenizer, max_length=128, unsupervised=False):
  if unsupervised:
    labels = None
    inputs = tokenizer(batch.masked, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
  else:
    labels = tokenizer(batch.fun, return_tensors='pt', padding=True, truncation=True, max_length=max_length)['input_ids']
    inputs = tokenizer(batch.masked, return_tensors='pt', padding='max_length', truncation=True, max_length=labels.size()[1])

  return inputs, labels

def keep_variant(variant: str):
  variant = variant.strip()
  output = True
  if len(variant) < 3 or variant in ('[CLS]','[PAD]',
  '[MASK]','[SEP]') or variant.startswith('#'):
    output = False
  else:
    variant_tag = nltk_to_upos(pos_tag([variant])[0][1])
    if variant_tag in ('PUNCT','ADP','DET','PRON','NUM',
    'PART','CCONJ','SYM','X','ADV'):
      output = False
  # print(variant, output)
  return output

def make_predictions(batch_iter, model, tokenizer, verbose=False, k=0):
  model.eval()

  predicted_sentences = []
  predicted_words = []

  with tt.no_grad():
    if verbose:
      print('Making predictions...')
      iterator = tqdm_notebook(batch_iter, total=len(batch_iter))
    else:
      iterator = batch_iter
    for batch in iterator:
      inputs, targets = tokenize_batch(batch, tokenizer, unsupervised=True)

      if k:
        output = model(**inputs)['logits'].topk(dim=2, k=k).indices.detach().numpy()
      else:
        output = model(**inputs)['logits'].argmax(dim=2).detach().numpy()

      # for sent in output:
      #   decoded_sent = tokenizer.decode(sent)
      #   # remove special tokens:
      #   decoded_sent = re.sub(r'^\[CLS\]', '', decoded_sent)
      #   decoded_sent = re.sub(r'\[SEP\]\s*?(\[PAD\]\s*?)*$', '', decoded_sent)
      #   decoded_sent = decoded_sent.strip()
      #   predicted_sentences.append(decoded_sent)

      masked_ids = inputs['input_ids'].numpy() == tokenizer.mask_token_id

      for sent_id, sent in enumerate(masked_ids):
        masked_seq = []
        for token_id, token in enumerate(sent):
          if token:
            masked_seq.append(output[sent_id][token_id])
        
        if k:
          masked_seq = np.array(masked_seq).transpose()
          predicted_span = tokenizer.batch_decode(masked_seq)
          ## filter predicted punctuation
          predicted_span = [variant for variant in predicted_span if keep_variant(variant)]
          # print(predicted_span)
        else:
          predicted_span = tokenizer.decode(masked_seq)
        predicted_words.append(predicted_span)
  return predicted_words
  # return predicted_sentences, predicted_words

def get_sent_probabilities(sents, bert_model, bert_tokenizer,
                           verbose=False):
  softmax = tt.nn.Softmax(dim=2)
  output_seqs = []

  if verbose:
    sents = tqdm_notebook(sents, total=len(sents))

  for sent in sents:
    inputs = bert_tokenizer([sent], return_tensors='pt',
                          padding=True, truncation=True)['input_ids']
    tok_sent = inputs[0].tolist()
    
    variants = []

    for token_id in range(inputs.shape[1]):
      variant = copy(tok_sent)
      variant[token_id] = 103
      variants.append(variant)
    
    variants_tensor = tt.LongTensor(variants)
    variants_out = bert_model(variants_tensor)['logits']

    tok_sent_probas = []

    for (i, tensor), right_token_id in zip(enumerate(softmax(variants_out)), tok_sent):
      token_tensor = tensor[i]
      proba = token_tensor[right_token_id].item()
      tok_sent_probas.append(proba)
    
    tokens = [bert_tokenizer.ids_to_tokens[elem] for elem in tok_sent]

    output_seqs.append([(token, round(proba, 4)) for token, proba in zip(tokens, tok_sent_probas)])
  return output_seqs

def process_lm_probas_output(sents_with_probas, verbose=False):
  output_sents = []

  if verbose:
    sents_with_probas = tqdm_notebook(sents_with_probas, total=len(sents_with_probas))

  for sent in sents_with_probas:
    output_sent = []
    output_probas = []
    for token_id, (token, proba) in enumerate(sent):
      if not token.startswith('##') and not token in ('[CLS]','[PAD]','[SEP]'):
        token_id += 1
        while token_id < len(sent):
          if sent[token_id][0].startswith('##'):
            token += sent[token_id][0][2:]
            token_id += 1
          else:
            break
        output_sent.append(token)
        output_probas.append(proba)
    
    ## здесь происходит pos-tagging - очевидно, недолгий и средствами nltk:
    output_sent = [(token, nltk_to_upos(tag)) for token, tag in pos_tag(output_sent)]
    
    output_sents.append([(token, tag, proba) for (token, tag), proba in zip(output_sent, output_probas)])
  
  # print(output_sents)
  return output_sents
