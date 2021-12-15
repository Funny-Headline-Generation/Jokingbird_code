import nltk
# import stanza
# import spacy_udpipe
import spacy

from functools import lru_cache

from utils import ud_to_wn, nltk_to_upos

class Tokenizer:
    '''
    Класс, позволяющий осуществить
    токенизацию различными библиотеками
    '''
    def __init__(self, method='spacy_upos'):
        self.method = method
        self.pipeline = None

        # if method == 'stanford_upos':
        #     ## токенизировать и лемматизировать будем стэнфордским парсером
        #     self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos')
        # elif method == 'udpipe_upos':
        #     ## токенизировать и лемматизировать будем UDPipe:
        #     self.pipeline = spacy_udpipe.load("en")
        if method == 'spacy_upos':
            ## токенизировать и лемматизировать будем SpaCy
            self.pipeline = spacy.load("en_core_web_sm")
        elif method == 'nltk':
            self.pipeline = nltk.stem.WordNetLemmatizer().lemmatize
    
    def __call__(self, text, include_orig_tokens=False):
        '''
        Перегрузка оператора () для токенизации текста
        '''
        if self.method == 'nltk':
            return self.nltk_tokenize(text, include_orig_tokens)
        elif self.method == 'stanford_upos':
            return self.stanford_upos_tokenize(text, include_orig_tokens)
        elif self.method == 'udpipe_upos':
            return self.udpipe_upos_tokenize(text. include_orig_tokens)
        elif self.method == 'spacy_upos':
            return self.spacy_upos_tokenize(text, include_orig_tokens)
    
    def nltk_lemmatize_sent(self, sent):
        sent = [(word, nltk_to_upos(tag)) for word, tag in sent]
        sent = [f'{self.pipeline(word, ud_to_wn(tag))}_{tag}' for word, tag in sent]
        return sent
    
    def nltk_tokenize(self, text, include_orig_tokens=False):
        ## токенизируем при помощи nltk
        text = nltk.sent_tokenize(text)
        text = [nltk.word_tokenize(sent) for sent in text]
        text = nltk.pos_tag_sents(text)
        lemmas = [self.nltk_lemmatize_sent(sent) for sent in text]
        if include_orig_tokens:
            tokens = [[f"{token}_{tag}" for token, tag in sent] for sent in text]
            return lemmas, tokens
        return lemmas
    
    # def stanford_upos_tokenize(self, text, include_orig_tokens=False):
    #     text = self.pipeline(text)
    #     lemmas = [[f"{token.lemma}_{token.upos}" for token in sent.words] for sent in text.sentences]

    #     if include_orig_tokens:
    #         tokens = [[f"{token.text}_{token.xpos}" for token in sent.words] for sent in text.sentences]
    #         return lemmas, tokens
    #     else:
    #         return lemmas
    
    # def udpipe_upos_tokenize(self, text, include_orig_tokens=False):
    #     ## токенизируем и лемматизируем при помощи udpipe
    #     text = self.pipeline(text)
    #     lemmas = [[f"{token.lemma}_{token.upos}" for token in sent.words] for sent in text.sentences]

    #     if include_orig_tokens:
    #         tokens = [[f"{token.text}_{token.xpos}" for token in sent.words] for sent in text.sentences]
    #         return lemmas, tokens
    #     else:
    #         return lemmas

    def spacy_upos_tokenize(self, text, include_orig_tokens=False):
        ## токенизируем и лемматизиурем при помощи SpaCy
        sents = [self.pipeline(sent) for sent in nltk.sent_tokenize(text)]
        lemmas = [[f"{token.lemma_}_{token.pos_}" for token in sent] for sent in sents]

        if include_orig_tokens:
            tokens = [[f"{token.text}_{token.tag_}" for token in sent] for sent in sents]
            return lemmas, tokens
        else:
            return lemmas
