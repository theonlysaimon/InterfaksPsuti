from itertools import combinations
from collections import Counter
import np
import json
from pymystem3 import Mystem
from nltk import FreqDist
import string
import os ## работаем с файлами, значит, наверняка понадобится ос


ng_1_data = []
with open("data/ng_1.jsonlines", "r") as read_file:
    for line in read_file:
        ng_1_data.append(json.loads(line)) # json.loads считывает строку, в отличие от json.load

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

for item in ng_1_data[:10]:
    print ('Эталонные ключевые слова: ', item['keywords'])
    print ('Самые частотные слова: ',  keywords_most_frequent_with_stop_and_lemm (item['content'], 6, rus_stops))
    print ()