from itertools import combinations
from collections import Counter
import np
import json

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

for text in file_texts:
    print (get_kws (word_tokenize(text))) # функция принимает на вход список слов, а не строку, поэтому так

def preprocessing_general (input_text, stoplist):
    '''функция для предобработки текста; 
    на вход принимает строку с текстом input_text и список стоп-слов stoplist
    на выходе чистый список слов output'''
    ## лемматизируем майстемом и делаем strip каждого слова:
    output = [word.strip() for word in moi_analizator.lemmatize (input_text)] 
    ## убираем пунктуацию и стоп-слова:
    output = [word for word in output if word not in extended_punctuation and word not in stoplist]
    ## убираем слова, в которых вообще нет буквенных символов:
    output = [word for word in output if re.search ('[А-ЯЁа-яёA-Za-z]', word) != None]
    return output

for text in file_texts:
    print (get_kws (preprocessing_general(text, rus_stops)))