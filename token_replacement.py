from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch as tt

from gensim.models import KeyedVectors
from gensim import matutils
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

## Local imports:
from collocations import CollocateMatrix, CollocationMetric
from nn_utils import UnsupervisedBatchIterator, make_predictions

class TokenReplacer(ABC):
    pass

class TokenIDReplacer(TokenReplacer, ABC):
    @abstractmethod
    def replace_tokens(self, sents, replacement_ids):
        pass

class MaskedTokenReplacer(TokenReplacer, ABC):
    @abstractmethod
    def replace_tokens(self, masked_sentences):
        pass

class GensimCollocateReplacer(TokenIDReplacer):
    def __init__(self, gensim_model_path: str, colloc_matrix_path: str, dist_thresh=0.41,
    colloc_thresh=2.0, verbose=False, return_all=False,
    colloc_metric=None, **kwargs):
        self.gensim_model = KeyedVectors.load_word2vec_format(gensim_model_path, binary=True)
        self.colloc_metric = colloc_metric
        self.cmatrix = CollocateMatrix.load(colloc_matrix_path)

        if self.colloc_metric is not None:
            if type(self.cmatrix.metric) != type(self.colloc_metric):
                self.cmatrix.recalculate_metric(self.colloc_metric)
        else:
            self.colloc_metric = self.cmatrix.metric

        self.dist_thresh = dist_thresh
        self.colloc_thresh = colloc_thresh
        self.verbose = verbose
        self.return_all = return_all
    
    def most_distant_to_vec(self, word_vec,
    pos, max_count=50):
        to_dist = lambda x: 1 - (x + 1)/2
        
        # cosine distance:
        # dot product:
        dists = np.dot(self.gensim_model.vectors, word_vec)
        # get_norms:
        model_norms = np.sqrt((self.gensim_model.vectors ** 2).sum(axis=1))
        word_norm = np.sqrt((word_vec ** 2).sum())
        # normalise:
        dists /= model_norms * word_norm
        ## map cosines to (0,1) range
        ## where higher values indicate higher distance:
        dists = to_dist(dists)

        #print(dists.max(), dists.min())

        sorted_dist_ids = matutils.argsort(dists, reverse=True)
        
        word_distances = [
            (self.gensim_model.index2entity[word_id], float(dists[word_id]))
            for word_id in sorted_dist_ids \
                if self.gensim_model.index2entity[word_id].endswith(pos)
                    and float(dists[word_id]) > self.dist_thresh
        ]

        if max_count:
            word_distances = word_distances[:max_count]
        
        return word_distances
    
    def replace_tokens(self, sents, replacement_ids):
        changed_headlines = []
        new_words = []


        if self.verbose:
            print("Replacing words with the most semantically distanct collocates of their neighbours")
            iterator = tqdm(zip(sents, replacement_ids), total=len(sents))
        else:
            iterator = zip(sents, replacement_ids)

        for sent, to_replace in iterator:
            sent_changed_headlines = []
            sent_new_words = []

            for variant in to_replace:
                changed_headline = sent
                new_word = None

                span_start_id = variant[0]
                span_end_id = variant[-1]+1
                
                value = sent[span_start_id:span_end_id]

                ## Берём часть речи первого слова в span'е
                pos = value[0].split('_')[1]
                word_vector = np.mean(np.array([self.gensim_model[word]\
                    for word in sent[span_start_id:span_end_id]\
                    if word in self.gensim_model]),
                axis=0)

                ## Данный блок надо переписать - сначала берём другие коллокаты,
                ## а потом находим наиболее далёкие от них семантически слова
                if type(word_vector) == np.ndarray:
                    most_distant = self.most_distant_to_vec(word_vector,
                                                            pos, max_count=0)

                    for new_word_id, (new_word_variant, distance) in enumerate(most_distant):
                        cstrength = 0

                        ## Select strongest collocation key:
                        for i in range(1, self.cmatrix.n):
                            key = tuple(sent[span_start_id-i:span_start_id])
                            cstrength_key = self.cmatrix.collocation_strength(key, new_word_variant)
                            #print(key, new_word_variant, cstrength_key)
                            if cstrength_key is not None:
                                if cstrength_key > cstrength:
                                    cstrength = cstrength_key
                        
                        most_distant[new_word_id] = (new_word_variant, distance, cstrength)
                    
                    ## отсеиваем коллокации по colloc_thresh:
                    most_distant = [(new_word_variant, distance, cstrength) for new_word_variant, distance, cstrength in most_distant if cstrength > self.colloc_thresh]

                    if self.return_all:
                        # most_distant = [(word, d, s) for (word, d, s) in most_distant if s > 0]
                        #print(most_distant, new_word)
                        #print(1, self.return_all, type(new_word))
                        if most_distant:
                            new_word = most_distant
                            changed_headline = [sent[:span_start_id] + [new_word] + sent[span_end_id:] for (new_word, d, s) in most_distant]
                            #print(2, self.return_all, type(new_word))
                    else:
                        ## Выберем наилучшего кандидата на замену данного слова
                        # Прибавляем единицу на случай если коллокационная матрица не знает такого сочетания,
                        # чтобы ранжировать только по расстоянию:
                        candidates = [(w,d*(cs+1)) for w,d,cs in most_distant]

                        if candidates:
                            new_word = max(candidates, key=lambda x: x[1])[0]

                            changed_headline = sent[:span_start_id] + [new_word] + sent[span_end_id:]
                #print(3, self.return_all, type(new_word))
                sent_changed_headlines.append(changed_headline)
                sent_new_words.append(new_word)

            changed_headlines.append(sent_changed_headlines)
            new_words.append(sent_new_words)
        
        return changed_headlines, new_words

class BERTReplacer(MaskedTokenReplacer):
    def __init__(self, bert_model_path: str, bert_tokenizer: str,
                 verbose=False, k=0,
                 **kwargs):
        ## Загрузим базовую модель и токенизатор для неё:
        self.bert_model = tt.load(bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.verbose = verbose
        self.return_all = False
        self.k = k

    def replace_tokens(self, masked_sentences):
        ## Превращаем двумерный список вариантов изменения каждого предложения в одномерный:
        masked_sentences_flat = list(chain(*masked_sentences))
        masked_sentences_flat = [s.lower().replace('[mask]','[MASK]').strip() for s in masked_sentences_flat]

        batch_iter = UnsupervisedBatchIterator(masked_sentences_flat)
        # changed_headlines_flat, new_words_flat = make_predictions(batch_iter, self.bert_model,
        # self.tokenizer, self.verbose)
        new_words_flat = make_predictions(batch_iter, self.bert_model,
        self.tokenizer, self.verbose, k=self.k)

        ## "Заворачиваем" обратно в двумерный список:
        # changed_headlines_iter, new_words_iter = iter(changed_headlines_flat), iter(new_words_flat)
        # changed_headlines, new_words = [], []
        # for sent in masked_sentences:
        #     changed_headline_variants = []
        #     new_words_sent = []
        #     for i in range(len(sent)):
        #         changed_headline_variants.append(next(changed_headlines_iter))
        #         new_words_sent.append(next(new_words_iter))
        #     changed_headlines.append(changed_headline_variants)
        #     new_words.append(new_words_sent)
        new_words_iter = iter(new_words_flat)
        changed_headlines, new_words = [], []
        for sent in masked_sentences:
            changed_headline_variants = []
            new_words_sent = []
            for variant in sent:
                if not self.k:
                    new_word = next(new_words_iter)
                    changed_headline_variants.append(variant.replace('[MASK]', new_word))
                    new_words_sent.append(new_word)
                else:
                    top_k_words = next(new_words_iter)
                    # print(top_k_words)
                    changed_headline_variants.append([variant.replace('[MASK]', new_word) for new_word in top_k_words])
                    new_words_sent.append(top_k_words)
            changed_headlines.append(changed_headline_variants)
            new_words.append(new_words_sent)

        return changed_headlines, new_words