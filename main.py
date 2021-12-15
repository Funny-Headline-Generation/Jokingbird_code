import argparse
import sys
import re

import numpy as np
import pandas as pd
import torch as tt
from time import time

from itertools import chain
from functools import lru_cache

from abc import ABC, abstractmethod
from copy import copy

from gensim.models import KeyedVectors
from gensim import matutils

from transformers import BertForMaskedLM, BertTokenizer

from tqdm import tqdm

## Local imports:
import collocations
from collocations import CollocateMatrix, CollocationMetric
from preprocessing import Tokenizer
from nn_utils import UnsupervisedBatchIterator, make_predictions
from utils import inflect, last_underscore_split
from new_colbert_predict.predict import predict_score
from token_selection import TokenSelector, CollocateTokenSelector, BERTTokenSelector
from token_replacement import TokenReplacer, TokenIDReplacer, MaskedTokenReplacer
from token_replacement import GensimCollocateReplacer, BERTReplacer

## TO DO:
## Встраивание предсказанного слова в предложение в GensimCollocateReplacer- Done
## - Интерфейс командной строки - загрузка из excel/csv с вышрузкой в excel/csv с нужной комбинацией моделей - In progress
## - Добавить интерфейс для обучения коллокационной матрицы и bert'а - если успею
## - Другие модели выбора слова для замены (вся коллокация, именованная сущность, рандомное слово) - если успею
## - Обучить коллокационную матрицу на cased data - если успею
## - Зафайнтюнить берт на датасете юмора с маскированием всех не-служебных слов

class Humourizer:
    def __init__(self,
                 token_selector: TokenSelector,
                 token_replacer: TokenReplacer,
                 verbose: bool,
                 score: bool):
        ## Используем инъекцию зависимостей:
        ## То как именно будут процессится
        ## предложения зависит от такого, какие
        ## объекты будут занимать поля
        ## token_selector,
        ## token_replacer
        self.token_selector = token_selector
        self.token_replacer = token_replacer
        self.verbose = verbose
        self.score = score
    
    def construct_sentence(self, new_word, replacement_id, sent_tokens):
        lemma, upos = last_underscore_split(new_word)

        ## Форму (пока) определяем по первому токену заменённого промежутка
        orig_token, orig_xpos = last_underscore_split(sent_tokens[replacement_id[0]])

        new_word_flexed = inflect(lemma, orig_xpos)

        ## "Собираем" предложение обратно
        tokens = [i[:i.rfind('_')] for i in sent_tokens]
        sent_variant = tokens[:replacement_id[0]]
        sent_variant += [new_word_flexed]
        sent_variant += tokens[replacement_id[-1]+1:]
        sent_variant = ' '.join(sent_variant)

        return sent_variant
    
    def join_new_sentences_upos_(self, headlines, replacement_ids, new_words):
        output_sentences = []

        for sent, sent_repl_ids, sent_new_words in zip(headlines,
        replacement_ids, new_words):
            sent_variants = []

            sent_lemmas, sent_tokens = self.token_replacer.cmatrix.tokenizer(sent,
            include_orig_tokens=True)

            sent_lemmas = [word for s in sent for word in s]
            sent_tokens = [word for s in sent_tokens for word in s]

            for replacement_id, new_word in zip(sent_repl_ids, sent_new_words):
                if new_word:
                    if type(new_word) == str:
                        sent_variant = self.construct_sentence(new_word, replacement_id, sent_tokens)
                    elif type(new_word) == list:
                        sent_variant = []
                        for new_word_variant, d, s in new_word:
                            sent_variant.append(self.construct_sentence(new_word_variant, replacement_id, sent_tokens))
                else:
                    sent_variant = sent

                sent_variants.append(sent_variant)
            
            output_sentences.append(sent_variants)
        return output_sentences
    
    def vandalize_headlines(self, headlines, return_pandas=False):
        ## Убираем источник:
        headlines = headlines.apply(lambda s: re.sub(' ?\- ?([Tt]he [Nn]ew [Yy]ork [Tt]imes|[Bb]reitbart)$','',s))

        ## Выбираем слова для замены
        tokenized_headlines, replacement_ids, replaced_words, masked_headlines = self.token_selector.select_tokens(headlines)
        # ## Маскируем слова для замены (для наглядности)
        # masked_sentences, masked_words = self.token_selector.mask_sentences(self, headlines, replacement_ids)
        ## Заменяем слова
        if isinstance(self.token_replacer, TokenIDReplacer):
            changed_headlines, new_words = self.token_replacer.replace_tokens(tokenized_headlines, replacement_ids)
            ## собираем предложение обратно:
            if type(self.token_replacer) == GensimCollocateReplacer:
                changed_headlines = self.join_new_sentences_upos_(headlines, replacement_ids, new_words)

        elif isinstance(self.token_replacer, MaskedTokenReplacer):
            changed_headlines, new_words = self.token_replacer.replace_tokens(masked_headlines)

        
        if return_pandas:
            outp = []

            if type(self.token_replacer) == GensimCollocateReplacer and self.token_replacer.return_all:
                for sent_id in range(len(tokenized_headlines)):
                    # print(len(masked_headlines[sent_id]), len(replacement_ids[sent_id]),
                    # len(replaced_words[sent_id]), len(changed_headlines[sent_id]), len(new_words[sent_id]))
                    for masked, index, replaced, changed, new in zip(masked_headlines[sent_id],
                    replacement_ids[sent_id],
                    replaced_words[sent_id],
                    changed_headlines[sent_id],
                    new_words[sent_id]):
                        if type(new) == list:
                            for (new_word, dist, cstrength), changed_sent in zip(new, changed):
                                outp.append({
                                    'headline':headlines.iloc[sent_id],
                                    'masked': masked,
                                    'tokenized': tokenized_headlines[sent_id],
                                    'span_index': index,
                                    'span': replaced,
                                    'predicted': changed_sent,
                                    'new span': new_word,
                                    'collocation_strength': cstrength,
                                    'semantic_distance': dist
                            })
                        else:
                            outp.append({
                                'headline':headlines.iloc[sent_id],
                                'masked': masked,
                                'tokenized': tokenized_headlines[sent_id],
                                'span_index': index,
                                'span': replaced,
                                'predicted': changed,
                                'new span': new,
                                'collocation_strength': None,
                                'semantic_distance': None
                            })
                outp = pd.DataFrame(outp)

                if self.score:
                    print('Scoring examples...')
                    scores = predict_score(outp, 'predicted',
                        'colbert_predict/colbert',
                        batch_size=16)
                    outp['predicted_score'] = scores
                    outp = outp.sort_values(by=['predicted_score'], ascending=False).drop_duplicates(['headline'])
                
                return outp
            elif type(self.token_replacer) == BERTReplacer and self.token_replacer.k:
                for sent_id in range(len(tokenized_headlines)):
                    for masked, index, replaced, changed, new in zip(masked_headlines[sent_id],
                    replacement_ids[sent_id],
                    replaced_words[sent_id],
                    changed_headlines[sent_id],
                    new_words[sent_id]):
                        # print(changed)
                        for changed_sent, new_word in zip(changed, new):
                            headline = headlines.iloc[sent_id]
                            if headline.strip().lower().replace(' ','').replace('_','') != changed_sent.strip().lower().replace(' ','').replace('_',''):
                                # print(headline, changed_sent)
                                outp.append({
                                    'headline': headline,
                                    'masked': masked,
                                    'tokenized': tokenized_headlines[sent_id],
                                    'span_index': index,
                                    'span': replaced,
                                    'predicted': changed_sent,
                                    'new span': new_word
                                })
            else:
                for sent_id in range(len(tokenized_headlines)):
                    # print(sent_id)
                    for masked, index, replaced, changed, new in zip(masked_headlines[sent_id],
                    replacement_ids[sent_id],
                    replaced_words[sent_id],
                    changed_headlines[sent_id],
                    new_words[sent_id]):
                        outp.append({
                            'headline':headlines.iloc[sent_id],
                            'masked': masked,
                            'tokenized': tokenized_headlines[sent_id],
                            'span_index': index,
                            'span': replaced,
                            'predicted': changed,
                            'new span': new
                        })
                
            outp = pd.DataFrame(outp)

            if self.score:
                print('Scoring examples...')
                scores = predict_score(outp, 'predicted',
                    'new_colbert_predict/colbert-trained',
                    batch_size=16)
                outp['predicted_score'] = scores
                outp = outp.sort_values(by=['predicted_score'], ascending=False).drop_duplicates(['headline'])
                #outp = outp.sort_values(by=['predicted_score'], ascending=False)

            return outp
                    
        return (tokenized_headlines, masked_headlines, replacement_ids,
        replaced_words, changed_headlines, new_words)


def test_collocate_gensim():
    token_selector = CollocateTokenSelector(colloc_matrix_path='CM_SpaCy_truecased')
    token_replacer = GensimCollocateReplacer(colloc_matrix_path='CM_SpaCy_truecased',
                                            gensim_model_path='gensim_models/udpipe_wikipedia/model.bin')
    sentences = ['i saw donald trump on wall street',
    'great britain to announce new prime minister']
    humourizer = Humourizer(token_selector, token_replacer, verbose=False, keep_case=True)
    return humourizer.vandalize_headlines(sentences,
    return_pandas=True)

def test_collocate_bert():
    token_selector = CollocateTokenSelector(colloc_matrix_path='CM_SpaCy_truecased')
    token_replacer = BERTReplacer('bert_masked_lm_full_model',
                                  'bert-large-uncased-whole-word-masking')
    sentences = ['i saw donald trump on wall street',
    'great britain to announce new prime minister']
    humourizer = Humourizer(token_selector, token_replacer, verbose=False, keep_case=True)
    return humourizer.vandalize_headlines(sentences,
    return_pandas=True)

def main(input_file: str, output_file: str, verbose: bool,
         keep_case: bool, token_selector: str, token_replacer: str,
         score: bool, colloc_metric: str,
         **kwargs):
    ext_in = input_file[input_file.rfind('.')+1:].lower()
    ext_out = output_file[output_file.rfind('.')+1:].lower()

    if ext_in not in ('csv', 'xlsx'):
        raise ValueError(f"Input extension .{ext_in} not supported")
    
    if ext_out not in ('csv', 'xlsx'):
        raise ValueError(f"Output extension .{ext_out} not supported")
    
    if ext_in == 'csv':
        df1 = pd.read_csv(input_file)
    elif ext_in == 'xlsx':
        df1 = pd.read_excel(input_file, engine='openpyxl')
    
    df1 = df1.dropna(subset=['headline'])

    if not keep_case:
        df1['headline'] = df1['headline'].str.lower()
    
    headlines = df1['headline']

    token_selector_cls = getattr(sys.modules[__name__], token_selector)

    if token_replacer == 'DistReplacer':
        token_replacer = 'GensimCollocateReplacer'

    token_replacer_cls = getattr(sys.modules[__name__], token_replacer)

    if colloc_metric is not None:
        colloc_metric = getattr(collocations, colloc_metric)()
    else:
        colloc_metric_cls = None

    time_now = time()
    token_selector = token_selector_cls(**kwargs, verbose=verbose,
    colloc_metric=colloc_metric)
    print(f"Token selector ready, time - {time()-time_now} s")

    time_now = time()
    token_replacer = token_replacer_cls(**kwargs, verbose=verbose,
    colloc_metric=colloc_metric)
    print(f"Token replacer ready, time - {time()-time_now} s")

    humourizer = Humourizer(token_selector, token_replacer, verbose, score)

    df2 = humourizer.vandalize_headlines(headlines, return_pandas=True)

    # print(type(df2))

    df_out = df1.merge(df2, on='headline', how='left')

    if humourizer.score:
        df_out = df_out.sort_values(by=['predicted_score'],
                                    ascending=False)

    print(f"Saving to {output_file}")

    if ext_out == 'csv':
        df_out.to_csv(output_file)
    elif ext_out == 'xlsx':
        df_out.to_excel(output_file, engine='openpyxl')

if __name__ == '__main__':
    ## gensim_models/News_of_the_web_2017_lemmatized/2017.bin
    parser = argparse.ArgumentParser(description='''
    Vandalize news headlines by making humorous edits of them.
    News headlines should be contained in a column called 'headline' of a CSV or XLSX file.
    The current implementation outputs all possible variants produced by algorithms''')

    parser.add_argument('input_file', help='CSV or Excel file with string column headline')
    parser.add_argument('output_file', help='Path to CSV/Excel file with headline_vandalizer output')

    parser.add_argument('--silent', dest='verbose', action='store_false',
    help='Not to display information regarding progress of data processing')
    parser.add_argument('--keep_case', dest='keep_case', action='store_true',
    help='Do not lowercase input data')

    # подклассы TokenSelector - конкретные реализации алгоритмов выбора токенов:
    token_selector_list = [c.__name__ for c in TokenSelector.__subclasses__()]
    # только подклассы подклассов TokenReplacer - конкрентные реализации алгоритмов выбора токенов:
    token_replacer_list = [subcl.__name__ for c in TokenReplacer.__subclasses__() for subcl in c.__subclasses__()]
    # метрики коллокационной силы:
    colloc_metric_list = [c.__name__ for c in CollocationMetric.__subclasses__()]

    ## Model-specific args:
    parser.add_argument('--word_selector', dest='token_selector', default='CollocateTokenSelector', choices=token_selector_list,
    help="Which word selection algorithm to use")
    parser.add_argument('--word_replacer', dest='token_replacer', default='BERTReplacer',
    choices=['BERTReplacer','DistReplacer'],
    help="Which word replacement algorithm to use")

    parser.add_argument('--gensim_model_path', dest='gensim_model_path', default='gensim_models/udpipe_wikipedia/model.bin',
    help='Path to gensim model file (must be ib binary Word2Vec format). Required if you are going to use DistReplacer')

    ## TO DO: Дописать код, чтобы аргумент работал:
    parser.add_argument('--dist_thresh', dest='dist_thresh', type=float, default=0.4,
    help='Threshold for semantic distance when replacing words with distance-based replacer')

    parser.add_argument('--colloc_matrix_path', dest='colloc_matrix_path', default='CM_SpaCy_truecased',
    help='Path to folder with saved CollocateMatrix. Required if you are going to use CollocateTokenSelector and/or DistReplacer')
    parser.add_argument('--colloc_metric', dest='colloc_metric', default=None, choices=colloc_metric_list,
    help="Which collocation metric to use")

    ## TO DO: Дописать код, чтобы аргумент работал:
    parser.add_argument('--colloc_thresh', dest='colloc_thresh', type=float, default=3.0,
    help="Which threshold to use for selecting collocations")

    parser.add_argument('--keep_all', dest='return_all', action='store_true',
    help='Whether to keep all candidates in output of DistReplacer')

    parser.add_argument('--bert_model_path', dest='bert_model_path', default='bert_humicroedit',
    help="Path to saved BERT model. Required if you are going to use BERTReplacer")

    parser.add_argument('--bert_selector_model', dest='bert_selector_model', default='bert-large-uncased-whole-word-masking',
    help='Name of BERT model from transformers repository for BERTSelector')
    # Tokenizer for bert_masked_lm_full_model - bert-large-uncased-whole-word-masking
    # Tokenizer for bert_jokes - bert-base-uncased
    parser.add_argument('--bert_tokenizer', dest='bert_tokenizer', default='bert-large-uncased-whole-word-masking',
    help='Name of BERT tokenizer from transformers repository which was used for model training. Required if you are going to use BERTReplacer')
    parser.add_argument('--top_k', dest='k', type=int, default=0,
    help="Select top k predictions for bert-based model")

    parser.add_argument('--score', dest='score', action='store_true',
    help='Whether to score outputs with humour classifier and keep only best results')

    args = vars(parser.parse_args())

    main(**args)
    # args = {
    #     'input_file': 'hundred_lines_truecased.xlsx',
    #     'output_file': 'hundred_lines_truecased_processed_newsbody.xlsx',
    #     'verbose': True,
    #     'score': True,
    #     'keep_case': True,
    #     'token_selector': 'CollocateTokenSelector',
    #     'token_replacer': 'GensimCollocateReplacer',
    #     'gensim_model_path': 'gensim_models/udpipe_wikipedia/model.bin',
    #     'colloc_matrix_path': 'CM_SpaCy_newsbody',
    #     'return_all': True
    # }

    # main(**args)
