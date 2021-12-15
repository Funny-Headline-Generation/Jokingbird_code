import torch as tt

from abc import ABC, abstractmethod
from copy import copy

from tqdm import tqdm, tqdm_notebook
from transformers import BertForMaskedLM, BertTokenizer

from collocations import CollocateMatrix, CollocationMetric, func_POS
from nn_utils import get_sent_probabilities, process_lm_probas_output

class TokenSelector(ABC):
    '''
    Выбирает индексы токенов
    в предложении, которые нужно заменить
    '''

    @abstractmethod
    def select_tokens(self, sents):
        '''
        Возвращает список списков кортежей:
        Для каждого предложения
        Список индексов промежутков в нём,
        которые нужно заменить
        '''
        pass

class CollocateTokenSelector(TokenSelector):
    def __init__(self, colloc_matrix_path, colloc_thresh=2.0, direction='right', verbose=False,
    colloc_metric=None, **kwargs):
        '''
        Выбрать слова для замены в устойчивых коллокациях

        colloc_matrix_path - Путь к сохранённой коллокационной матрице
        '''

        self.colloc_metric = colloc_metric
        self.cmatrix = CollocateMatrix.load(colloc_matrix_path)

        if self.colloc_metric is not None:
            if type(self.colloc_metric) != type(self.cmatrix.metric):
                self.cmatrix.recalculate_metric(self.colloc_metric)
        else:
            self.colloc_metric = self.cmatrix.metric

        self.thresh = colloc_thresh
        self.direction = direction
        self.verbose = verbose
    
    def select_tokens(self, sents, mask_token='[MASK]'):
        ## Токенизируем, предполагая,
        ## что один заголовок - всегда одно предложение
        
        tokenized_sents = []
        masked_sents = []
        replacement_ids = []
        replaced_words = []


        if self.verbose:
            print("Selecting collocates to replace")
            iterator = tqdm(sents, total=len(sents))
        else:
            iterator = sents

        for sent in iterator:
            sent, sent_tokens = self.cmatrix.tokenizer(sent, include_orig_tokens=True)

            sent = [word for s in sent for word in s]
            sent_tokens = [word[:word.rfind('_')] for s in sent_tokens for word in s]

            sent_repl_ids = []
            sent_replaced_words = []
            masked_variants = []

            collocations = self.cmatrix.get_collocations(sent,
            thresh=self.thresh, tokenize=False)

            for idxs, collocation, strength in collocations:
                ## Берём правую часть коллокации (всегда одно слово)
                repl_id = idxs[-1]

                if (repl_id,) not in sent_repl_ids:
                    sent_repl_ids.append((repl_id,))
                    sent_replaced_words.append((collocation[-1],))

                    masked_variant = copy(sent_tokens)
                    masked_variant[repl_id] = mask_token
                    masked_variants.append(' '.join(masked_variant))
            
            tokenized_sents.append(sent)
            masked_sents.append(masked_variants)
            replacement_ids.append(sent_repl_ids)
            replaced_words.append(sent_replaced_words)
        
        return tokenized_sents, replacement_ids, replaced_words, masked_sents

class BERTTokenSelector(TokenSelector):
    def __init__(self, bert_selector_model, bert_tokenizer,
                  colloc_thresh,
                  verbose = False, **kwargs):
        self.bert_model = BertForMaskedLM.from_pretrained(bert_selector_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.proba_thresh = colloc_thresh
        self.verbose = verbose

    def select_tokens(self, sents):
        sents_with_probas = get_sent_probabilities(sents, self.bert_model, self.bert_tokenizer, self.verbose)
        sents_with_probas = process_lm_probas_output(sents_with_probas, self.verbose)

        if self.verbose:
            sents_with_probas = tqdm_notebook(sents_with_probas, total=len(sents_with_probas))
        
        tokenized_headlines = [[token+"_"+tag for token, tag, proba in sent] for sent in sents_with_probas]
        
        replacement_ids, replaced_words, masked_headlines = [], [], []
        for sent_id, sent in enumerate(sents_with_probas):
            only_tokens = [token for token, tag, proba in sent]
            sent_repl_ids = []
            sent_replaced_words = []
            masked_variants = []
            for token_id, (token, tag, proba) in enumerate(sent):
                # print(token, tag, proba)
                if proba >= self.proba_thresh and not tag in func_POS:
                    # print(token, tag, proba)
                    sent_repl_ids.append((token_id,))
                    sent_replaced_words.append((token+'_'+tag,))
                    # Интересный факт - если заменить token_id+1 на token_id (т.е. вставлять слово перед сильным словом)
                    # - Выходит довольно смешно
                    masked_variants.append(' '.join(only_tokens[:token_id]+['[MASK]']+only_tokens[token_id+1:]))
            replacement_ids.append(sent_repl_ids)
            replaced_words.append(sent_replaced_words)
            masked_headlines.append(masked_variants)
        
        # print(tokenized_headlines[:5], replacement_ids[:5], replaced_words[:5], masked_headlines[:5])
        # print(len(tokenized_headlines))
        return tokenized_headlines, replacement_ids, replaced_words, masked_headlines
