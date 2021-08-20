from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from collections import Counter
import np
import json
from pymystem3 import Mystem
from nltk import FreqDist
import string
import os 
import re
    
ng_1_data = []
with open("D:\\Desktop\\Study\\GitHub\\InterfaksPsuti\\data\\ng.jsonlines", "r", encoding="utf8") as read_file:
    for line in read_file:
        ng_1_data.append(json.loads(line))

with open ('D:\\Desktop\\Study\\GitHub\\InterfaksPsuti\\data\\stop_ru.txt', 'r', encoding="utf8") as stop_file:
    rus_stops = [word.strip() for word in stop_file.readlines()] 

def get_kws(text, top=6, window_size=5, random_p=0.1):

    vocab = set(text)
    word2id = {w:i for i, w in enumerate(vocab)}
    id2word = {i:w for i, w in enumerate(vocab)}
    # преобразуем слова в индексы для удобства
    ids = [word2id[word] for word in text]

    # создадим матрицу совстречаемости
    m = np.zeros((len(vocab), len(vocab)))

    # пройдемся окном по всему тексту
    for i in range(0, len(ids), window_size):

        window = ids[i:i+window_size]
        # добавим единичку всем парам слов в этом окне
        for j, k in combinations(window, 2):
            # чтобы граф был ненаправленный 
            m[j][k] += 1
            m[k][j] += 1
    
    # нормализуем строки, чтобы получилась вероятность перехода
    for i in range(m.shape[0]):
        s = np.sum(m[i])
        if not s:
            continue
        m[i] /= s
    
    # случайно выберем первое слова, а затем будет выбирать на основе полученых распределений
    # сделаем так 5 раз и добавим каждое слово в счетчик
    # чтобы не забиться в одном круге, иногда будет перескакивать на случайное слово
    
    c = Counter()
    # начнем с абсолютного случайно выбранного элемента
    n = np.random.choice(len(vocab))
    for i in range(500): # если долго считается, можно уменьшить число проходов
        
        # c вероятностью random_p 
        # перескакиваем на другой узел
        go_random = np.random.choice([0, 1], p=[1-random_p, random_p])
        
        if go_random:
            n = np.random.choice(len(vocab))
        ### 
        n = take_step(n, m)
        # записываем узлы, в которых были
        c.update([n])
    
    # вернем топ-N наиболее часто встретившихся сл
    return [id2word[i] for i, count in c.most_common(top)]

def take_step(n, matrix):
    rang = len(matrix[n])
    # выбираем узел из заданного интервала, на основе распределения из матрицы совстречаемости
    if np.any(matrix[n]):
        next_n = np.random.choice(range(rang), p=matrix[n])
    else:
        next_n = np.random.choice(range(rang))
    return next_n

extended_punctuation = string.punctuation + '—»«...'

def passed_filter (some_word, stoplist):
    some_word = some_word.strip()
    if some_word in extended_punctuation:
        return False
    elif some_word in stoplist:
        return False
    elif re.search ('[А-ЯЁа-яёA-Za-z]', some_word) == None:
        return False
    return True

moi_analizator = Mystem()

def keywords_most_frequent_with_stop_and_lemm (some_text, num_most_freq, stoplist):
    lemmatized_text = [word for word in moi_analizator.lemmatize(some_text.lower()) 
                       if passed_filter(word, stoplist)]
    return [word_freq_pair[0] for word_freq_pair in FreqDist(lemmatized_text).most_common(num_most_freq)]

def preprocess_for_tfidif (some_text):
    lemmatized_text = moi_analizator.lemmatize(some_text.lower())
    return (' '.join(lemmatized_text)) # поскольку tfidf векторайзер принимает на вход строку, 
    #после лемматизации склеим все обратно

def produce_tf_idf_keywords (some_texts, number_of_words):
    make_tf_idf = TfidfVectorizer (stop_words=rus_stops)
    texts_as_tfidf_vectors=make_tf_idf.fit_transform(preprocess_for_tfidif(text) for text in some_texts)
    id2word = {i:word for i,word in enumerate(make_tf_idf.get_feature_names())} 

    for text_row in range(texts_as_tfidf_vectors.shape[0]): 
        ## берем ряд в нашей матрице -- он соответстует тексту:
        row_data = texts_as_tfidf_vectors.getrow(text_row)
        ## сортируем в нем все слова: 
        words_for_this_text = row_data.toarray().argsort() 
        ## берем число слов с конца, равное number_of_words 
        top_words_for_this_text = words_for_this_text [0, :-1*(number_of_words+1):-1]
        ## печатаем результат
        print([id2word[w] for w in top_words_for_this_text])


for item in ng_1_data[:10]:
    print ('Эталонные ключевые слова: ', item['keywords'])
    print ('Самые частотные слова: ',  keywords_most_frequent_with_stop_and_lemm (item['content'], 6, rus_stops))
    print ()

"""
manual_keywords = [] ## сюда запишем все ключевые слова, приписанные вручную
full_texts = [] ## сюда тексты

for item in ng_1_data:
    manual_keywords.append(item['keywords'])
    full_texts.append(item['content'])

produce_tf_idf_keywords (full_texts[:20], 6)
manual_keywords [:20]
"""