from preprocessing import Tokenizer
from itertools import chain
from functools import lru_cache

class TokenSelector:
    '''
    Выбирает индексы токенов
    в предложении, которые нужно заменить
    '''
    def __init__(self):
        pass

    def select_tokens(self, sents):
        '''
        Возвращает список списков кортежей:
        Для каждого предложения
        Список индексов промежутков в нём,
        которые нужно заменить
        '''
        pass

class TokenReplacer:
    '''
    Заменяет определённые индексы
    токенов в предложении по определённому
    алгоритму
    '''
    def __init__(self):
        pass

    @lru_cache(maxsize=1)
    def mask_sentences(self, sents, replacement_ids):
        pass

    def replace_tokens(self, sents, replacement_ids):
        pass

class Humourizer:
    def __init__(self, tokenizer: Tokenizer,
                 token_selector: TokenSelector,
                 token_replacer: TokenReplacer):
        ## Используем инъекцию зависимостей:
        ## То как именно будут процессится
        ## предложения зависит от такого, какие
        ## объекты будут занимать поля
        ## tokenizer, token_selector,
        ## token_replacer
        self.tokenizer = tokenizer
        self.token_selector = token_selector
        self.token_replacer = token_replacer
    
    def vandalize_headlines(self, headlines):
        ## Токенизируем, предполагая,
        ## что один заголовок - всегда одно предложение
        headlines = [list(chain(*headline)) for headline in self.tokenizer(headline)]
        ## Выбираем слова для замены
        replacement_ids = self.token_selector.select_tokens(headlines)
        ## Маскируем слова для замены (для наглядности)
        masked_sentences, masked_words = self.token_selector.mask_sentences(self, headlines, replacement_ids)
        ## Заменяем слова
        changed_sentences, new_words = self.token_replacer.replace_tokens(self, headlines, replacement_ids)

        return masked_sentences, masked_words, changed_sentences, new_words
