import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#--- stemmer Porter ---
RVRE = re.compile(r'^(.*?[аеиоуыэюя])(.*)$')
PERFECTIVEGROUND = re.compile(r'((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$')
REFLEXIVE = re.compile(r'(с[яь])$')
ADJECTIVE = re.compile(r'(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$')
VERB = re.compile(r'((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить'
                  r'|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$')
NOUN = re.compile(r'(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию'
                  r'|ью|ю|ия|ья|я)$')
I = re.compile(r'и$')
PARTICIPLE = re.compile(r'((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$')
DERIVATIONAL = re.compile(r'(ость|ост)$')
P = re.compile(r'ь$')
NN = re.compile(r'нн$')
SUPERLATIVE = re.compile(r'(ейше|ейш)$')
NOT_LETTER = re.compile(r'[^a-яА-Яё]$')

#--- stop-symbols ---
NEWLINE_SYMBOLS = re.compile(r'-\n|\n|\t')
STOP_SYMBOLS = re.compile(r'\s(-|а|без|более|больше|будет|будто|бы|был|была|были|было|быть|в|вам|вас|вдруг|ведь|во'
                          r'|вот|впрочем|все|всегда|всего|всех|всю|вы|г|где|говорил|да|даже|два|для|до|еще|ж|же|жизнь'
                          r'|за|зачем|здесь|и|из|из-за|или|им|иногда|их|к|кажется|как|какая|какой|когда|конечно|которого'
                          r'|которые|кто|куда|ли|лучше|между|меня|мне|много|может|можно|мой|моя|него|нее|ней|нельзя|нет'
                          r'|ни|нибудь|никогда|ним|них|ничего|но|ну|о|об|один|он|она|они|опять|от|перед|по|под|после|потом'
                          r'|потому|почти|при|про|раз|разве|с|сам|свое|сказать|со|совсем|так|такой|там|тебя|тем|теперь'
                          r'|то|тогда|того|тоже|только|том|тот|три|тут|ты|у|уж|уже|хорошо|хоть|чего|человек|чем|через'
                          r'|что|чтоб|чтобы|чуть|эти|этого|этой|другой|его|ее|ей|ему|если|есть|мы|на|над|надо|наконец'
                          r'|нас|не|свою|себе|себя|сегодня|сейчас|сказал|сказала|этом|этот|эту|я)\s')
OTHER_SYMBOLS = re.compile(r'\.|\'|\"|!|\?|,|:|&|\*|@|#|№|\(|\)|\[|\]|\{|\}|\$|%|\^|\\|/|;|\<|\>|\+|\-|\=|\s\d+|\d+\s')

#--- variables ---
all_words = {}                       # все тексты, представленный одной строкой
dict_texts = {}                    # ключи - T1, T2... значения - текст файлов Т1, Т2 и тд


# функция запрашивает пути файлов для считывания и помещает весь текст в словарь dict_texts.
def input_data():
    print("Укажите пути (по умолчанию 'D:\Учёба\\7 семестр -\- Технологии обработки информации (Экз)\Лабораторная работа №3\')")
    i = 0
    while True:
        path_to_file = input("Укажите путь к файлу: ")
        if path_to_file.upper() == 'END':
            break
        text_file = get_text(path_to_file)
        if text_file:
            dict_texts['T' + str(i + 1)] = text_file
            i += 1
        else:
            break


# функция считывает тексты из файлов.
def get_text(path_to_file):
    text = ''
    while True:
        try:
            file = open(path_to_file, "r")

            for line in file:
                text += line
            file.close()
            return text

        except:
            path_to_file = input("Не найден указанный путь. Попробуйте еще раз, или введите 'END', чтобы завершить: " + "\n")
            if path_to_file.upper() == "END":
                break

    return False


# в функцию передается текст из файла в виде строки. Из нее удаляются все ненужные символы и возвращается отформатиро-
# ванный текст.
def delete_stop_symbols(text):
    text = text.lower()
    result = NEWLINE_SYMBOLS.sub('', text)
    result = OTHER_SYMBOLS.sub('', result)
    result = ' ' + result + ' '
    result = STOP_SYMBOLS.sub(' ', result)
    return result


# функция реализует алгоритм стемминга. Принимает слово, возвращает основу слова.
def stemming(word):
    word = word.lower()
    word = word.replace('ё', 'e')
    area = re.match(RVRE, word)

    if area is not None:
        PREFIX = area.group(1)
        RV = area.group(2)

        # step 1
        template = PERFECTIVEGROUND.sub('', RV, 1)
        if template == RV:
            RV = REFLEXIVE.sub('', RV, 1)
            template = ADJECTIVE.sub('', RV, 1)

            if template != RV:
                RV = template
                RV = PARTICIPLE.sub('', RV, 1)
            else:
                template = VERB.sub('', RV, 1)
                if template == RV:
                    RV = NOUN.sub('', RV, 1)
                else:
                    RV = template
        else:
            RV = template

        # step 2
        RV = I.sub('', RV, 1)

        # step 3
        RV = DERIVATIONAL.sub('', RV, 1)

        # step 4
        template = NN.sub('н', RV, 1)
        if template == RV:
            template = SUPERLATIVE.sub('', RV, 1)
            if template != RV:
                RV = template
                RV = NN.sub('н', RV, 1)
            else:
                RV = P.sub('', RV, 1)
        else:
            RV = template
        word = PREFIX + RV
    return word


# функция строит частотную матрицу (столбец - j, строка i)
def build_frequency_matrix(dict):
    # matrix = len(dict.keys()) * [[0].copy() * len(dict_texts)]
    matrix = [[0 for i in range(len(dict_texts))] for j in range(len(all_words))]
    i = 0
    print("Всего слов: " + str(len(all_words)))
    print(all_words)
    for word in sorted(list(all_words)):
        j = 0
        for T in dict_texts:
            for element in dict_texts[T]:
                if element == word:
                    matrix[i][j] += 1
            j += 1
        i += 1
    for el in matrix:
        print(el)
    return matrix


# функция сингулярного разложения матрицы
def get_SVD(matrix):
    a = np.matrix(matrix)
    u, w, vt = np.linalg.svd(a, full_matrices=True)
    _pow = 2
    _u = [[None for j in range(_pow)] for i in range(len(all_words))]
    # приводим к двумерному сингулярному разложению
    for i in range(len(all_words)):
        for j in range(_pow):
            _u[i][j] = round(u[i, j], 2) * -1
    vt = vt[:_pow]
    w = w[:_pow]
    return _u, w, vt


# --- start program ---
input_data()
# step 1 - удаляем все стоп символы, знаки препинания, пробельные символы и дт.
for text in dict_texts:
    dict_texts[text] = delete_stop_symbols(dict_texts[text])

# step 2
# 2.1. разбиваем всё на массивы слов.
for text in dict_texts:
    dict_texts[text] = dict_texts[text].strip() # удалить все пробельные символы в начале и в конце.
    dict_texts[text] = dict_texts[text].split(' ')
    # print(dict_texts[text])
# 2.2. проводим операцию стемминга.
# step 3 - помещаем в словарь all_words все слова. Оставляем только повторяющиеся.
i = 0
for T in dict_texts:
    j = 0
    for word in dict_texts[T]:
        dict_texts[T][j] = stemming(word)
        if dict_texts[T][j] in all_words.keys():        # если ключ есть, то увеличиваем на 1, если нет - создается новый
            all_words[dict_texts[T][j]] += 1
        else:
            all_words[dict_texts[T][j]] = 1
        j += 1
    i += 1
# print(all_words)
for key in list(all_words):         # т.к. нельзя трогать ключи просматривая словарь, нужно создать создать копию через list()
    if all_words[key] == 1:
        all_words.pop(key)

# step 4 - создаем частотную матрицу
fr_matrix = build_frequency_matrix(all_words)

# step 5 - производим двумерное сингулярное разложение
U, W, Vt = get_SVD(fr_matrix)
print("---------------------------------------")
for el in U:
    print(el)
print("---------------------------------------")
for el in W:
    print(el)
print("---------------------------------------")
for el in Vt:
    print(el)

# Кластеризация
def clusterisation(arr):
    colors = ['b', 'g', 'r']  # цвет маркеров
    markers = ['o', 'v', 's'] # форма маркеров
    plt.plot()
    X = arr
    k = 3
    clusters_dict = {}
    kmeans_model = KMeans(n_clusters = k).fit(X)
    print(kmeans_model.predict(X))
    kmeans_points = kmeans_model.predict(X)
    plt.plot()
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(X[i][0], X[i][1], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([-0.08, 1])
        plt.ylim([-0.08, 1])
    plt.show()
    for i in range(len(kmeans_points)):
        if clusters_dict.get(kmeans_points[i]) == None:
            clusters_dict[kmeans_points[i]] = []
            clusters_dict[kmeans_points[i]].append('T' + str(i + 1))
        else:
            clusters_dict[kmeans_points[i]].append('T' + str(i + 1))
    return clusters_dict


array_to_K_means =[[0] * 2 for i in range(Vt[0].size)]
for i in range(2):
    for j in range(Vt[0].size):
        array_to_K_means[j][i] = Vt[i,j] * (-1)

result = clusterisation(array_to_K_means)
for key in result:
    print("---" * 10)
    for el in result[key]:
        print(el)
# step 6 - кластеризация

# D:\Учёба\7 семестр -\- Технологии обработки информации (Экз)\Лабораторная работа №3\Testing text
