import pandas as pd
from collections import defaultdict, Counter

# chars = [
#         ["Моника", "Мон"],
#         ["Рэйчел", "Рейч"],
#         ["Чендлер", "Чэндлер", "Чен"],
#         ["Фиби", "Фибс"],
#         ["Росс"],
#         ["Джоуи", "Джои", "Джо"],
#     ]

def make_pandus(matrix, words):
    return pd.DataFrame(matrix, columns=words)

# самые частые слова из матрицы
def most_freq(matrix, words):
    df = make_pandus(matrix, words)
    most_frequent = df.sum(axis=0).sort_values(ascending=False).index.to_list()[:1][0]
    return 'самое частое слово: {}'.format(most_frequent)

# самые редкие слова из матрицы
def least_freq(matrix, words):
    df = make_pandus(matrix, words)
    #least_frequent = df.sum(axis=0).sort_values(ascending=True).index.to_list()[:1][0]
    least = df.columns[(df > 0).sum() == 1].to_list() #действуя как-то иначе мы вытащим лишьпо одному слову
    return least

# набор слов, которые есть во всех документах коллекции, с помощью матрицы
def all_docs(matrix, words):
    df = make_pandus(matrix, words)
    return df.columns[(df > 0).sum() == 165].to_list()

# кто из главных героев упонимается чаще всего с помощью матрицы
def freq_count(matrix, words, chars):
    df = make_pandus(matrix, words)
    freq_dict = {'char_name' : [], 'freq' : []}
    summa = df.values.sum()
    for char in chars:
        char_name = ' / '.join(char)
        freq = 0
        for namevariant in char:
            try:
                freq += df[namevariant.lower()].sum()
            except:
                continue
        freq = freq / summa
        freq_dict['char_name'].append(char_name)
        freq_dict['freq'].append(freq)
    freq_df = pd.DataFrame(freq_dict)
    #freq_df = freq_df.round({'freq' : 5})
    return freq_df.sort_values('freq', ascending = False).head(1)['char_name'].reset_index(drop = True)[0]

# самые частые слова из словаря
def most_common(inverted_dict):
    common = 0
    for key, val in inverted_dict.items():
        for i in val:
            if i[1] > common:
                common = i[1]
                word = key
    return word

# самые редкие слова из словаря
def less_common(inverted_dict):
    words = [] # можно аналогично самому частому искать самое редкое
    for key, val in inverted_dict.items():
        for i in val:
            if i[1] == 1:
                words.append(key)
    return words

# набор слов, которые есть во всех документах коллекции, с помощью словаря
def in_every(inverted_dict):
    return [i for i, val in inverted_dict.items() if len(val) == 165]

# кто из главных героев упонимается чаще всего с помощью словаря
def most_mentioned_char(inverted_dict, chars):
    freq = defaultdict(int)
    for i, person in enumerate(chars):
        for j in person: # одно из имён
            try:
                freq[i] += sum([inverted_dict[j.lower()][k][1] for k in range(len(inverted_dict[j.lower()]))])
            except:
                continue
    return chars[Counter(freq).most_common(1)[0][0]][0]

