from preprocess import prepare
from inverted_index import indexing_matrix, indexing_dict
from calculations import *

dir_data = 'friends-data'

chars = [
        ["Моника", "Мон"],
        ["Рэйчел", "Рейч"],
        ["Чендлер", "Чэндлер", "Чен"],
        ["Фиби", "Фибс"],
        ["Росс"],
        ["Джоуи", "Джои", "Джо"],
    ]

def main():
    lem_corpus = prepare(dir_data)
    matrix, words = indexing_matrix(lem_corpus)
    inverted_dict = indexing_dict(matrix, words)
    #print('матрица с обратным индексом:', matrix, 'словарь с обратным индексом', inverted_dict, sep='\n')
    print('счёт через матрицу:', most_freq(matrix, words),
          'редких слов очень много, все они встречаются по одному разу, отобразим первые 100: {}'.format(least_freq(matrix, words)[:100]),
          'список слов, которые встречаются во всех текстах:', all_docs(matrix, words),
          'самый часто упоминаемый персонаж:', freq_count(matrix, words, chars), sep='\n')
    print('счёт через словарь:', 'самое частое слово:'.format(most_common(inverted_dict)),
          'редких слов очень много, все они встречаются по одному разу, отобразим первые 100: {}'.format(less_common(inverted_dict)[:100]),
          'список слов, которые встречаются во всех текстах:', in_every(inverted_dict),
          'самый часто упоминаемый персонаж:', most_mentioned_char(inverted_dict, chars), sep='\n')

if __name__ == '__main__':
    main()

