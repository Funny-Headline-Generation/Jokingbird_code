import lemminflect

def two_way_map(l):
    '''Функция, позволяющая из списка уникальных элементов получить
    два отображения:
    
    индекс-элемент
    элемент-индекс'''
    id2elem = [i for i in l]
    elem2id = {id2elem[i]:i for i in range(len(id2elem))}
    return id2elem, elem2id

def with_reverse_map(func, *args, **kwargs):
    def add_reverse_map(*args, **kwargs):
        func_result = func(*args, **kwargs)
        return two_way_map(func_result)
    return add_reverse_map

def inflect(lemma, xpos):
    inflections = lemminflect.getInflection(lemma, xpos)
    if inflections:
        return inflections[0]
    else:
        return lemma

def last_underscore_split(string):
    a = string[:string.rfind('_')]
    b = string[string.rfind('_')+1:]
    return a, b

penn_to_ud = {'#': 'SYM', '$': 'SYM', "''": 'PUNCT', ',': 'PUNCT', '-LRB-': 'PUNCT', '-RRB-': 'PUNCT', '.': 'PUNCT',
':': 'PUNCT', 'AFX': 'ADJ', 'CC': 'CCONJ', 'CD': 'NUM', 'DT': 'DET', 'EX': 'PRON', 'FW': 'X', 'HYPH': 'PUNCT',
'IN': 'ADP', 'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ', 'LS': 'X', 'MD': 'VERB', 'NIL': 'X', 'NN': 'NOUN',
'NNP': 'PROPN', 'NNPS': 'PROPN', 'NNS': 'NOUN', 'PDT': 'DET', 'POS': 'PART', 'PRP': 'PRON', 'PRP$': 'DET',
'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'RP': 'ADP', 'SYM': 'SYM', 'TO': 'PART', 'UH': 'INTJ', 'VB': 'VERB',
'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB', 'WDT': 'DET', 'WP': 'PRON',
'WP$': 'DET', 'WRB': 'ADV', '``': 'PUNCT'}

def nltk_to_upos(tag):
    if tag in penn_to_ud:
        return penn_to_ud[tag]
    else:
        return 'PUNCT'

wordnet_tags = {
    'ADJ': 'a',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v',
    'PROPN': 'n'
}

def ud_to_wn(tag):
    if tag in wordnet_tags:
        return wordnet_tags[tag]
    else:
        return 'n'