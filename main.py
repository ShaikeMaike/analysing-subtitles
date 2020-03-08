import os
import csv
import collections
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# return a dictionary with the bad word and the numbers of appearance
def extract_bad_words(year):
    with open(year, 'r',  encoding="utf-8") as f:
        text = f.read()

    f = open("bad-words.txt", "r")
    words_text = f.read()
    words_text = words_text.split("\n")

    words_text = set(words_text)
    dic = dict()

    for sentence in text.split("\n"):
        words_in_sentence = set(sentence.split())
        intersec = words_in_sentence.intersection(words_text)
        if intersec:
            for word in intersec:
                if word in dic.keys():
                    dic[word] += 1
                else:
                    dic[word] = 1
    # return dic.items()
    return dic


def find_result_for_all_decades():

    decade = 1930
    result_for_all_decades = dict()
    while decade != 2020:
        result = extract_bad_words(os.path.join('decades', str(decade)))
        if len(result.keys()) == 0:
            decade += 10
            continue

        result_for_all_decades[str(decade)] = result
        decade += 10
    return result_for_all_decades


def graph_for_number_of_words(number_of_words, kind):
    word_counter = collections.Counter(number_of_words)
    lst = word_counter.most_common(10)
    df = pd.DataFrame(lst, columns=[kind, 'Number'])
    df.plot.bar(x=kind, y='Number', title="number of offensive words in each {}".format(kind))
    plt.show()


def graph_for_percentage_of_words(number_of_words, kind):
    word_counter = collections.Counter(number_of_words)
    lst = word_counter.most_common(10)
    df = pd.DataFrame(lst, columns=[kind, 'Number'])
    df.plot.bar(x=kind, y='Number', title="percentage of offensive words in each {}".format(kind))
    plt.show()


def generate_bad_words_by_decades():

    decade = 1930
    number_of_words = dict()
    with open('result-decade.csv', mode='w') as csv_file:

        while decade != 2020:
            result = extract_bad_words(os.path.join('decades', str(decade)))

            if len(result.keys()) == 0:
                decade += 10
                continue
            fieldnames = ['decade', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'decade': decade, 'bad word': res[0], 'number of appearance': res[1]})

            number_of_words[str(decade)] = len(result.keys())

            word_counter = collections.Counter(result)
            lst = word_counter.most_common(10)
            df = pd.DataFrame(lst, columns=['Word', 'Count'])
            df.plot.bar(x='Word', y='Count', title=str(decade))
            plt.show()

            decade += 10

        graph_for_number_of_words(number_of_words, 'Decade')

    csv_file.close()


def find_num_all_word(path):
    with open(path, 'r',  encoding="utf-8") as f:
        text = f.read()

    counter = 0
    for sentence in text.split("\n"):
            counter += len(sentence)
    return counter


def generate_bad_words_by_score():
    score = 0
    number_of_words = dict()
    with open('result-score.csv', mode='w') as csv_file:

        while score != 11:
            result = extract_bad_words(os.path.join('score', 'score_' + str(score)))
            fieldnames = ['score', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'score': score, 'bad word': res[0], 'number of appearance': res[1]})

            all_words = find_num_all_word(os.path.join('score', 'score_' + str(score)))
            number_of_words[str(score)] =(len(result.keys()) / all_words) * 100

            word_counter = collections.Counter(result)
            lst = word_counter.most_common(10)
            df = pd.DataFrame(lst, columns=['Word', 'Count'])
            df.plot.bar(x='Word', y='Count')
            plt.show()

            score += 1
        graph_for_percentage_of_words(number_of_words, 'Score')

    csv_file.close()


def generate_bad_words_by_genres():
    number_of_words = dict()
    with open('result-genre.csv', mode='w') as csv_file:

        for genre in ['Western', 'War', 'Thriller', 'Science', 'Romance', 'Mystery', 'Music',
                      'Horror', 'History', 'Foreign', 'Fantasy', 'Family', 'Drama', 'Documentary',
                      'Crime', 'Comedy', 'Animation', 'Adventure', 'Action']:
            result = extract_bad_words(os.path.join('genres', genre))
            fieldnames = ['genre', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'genre': genre, 'bad word': res[0], 'number of appearance': res[1]})

            all_words = find_num_all_word(os.path.join('genres', genre))
            number_of_words[genre] = (len(result.keys()) / all_words) * 100

            word_counter = collections.Counter(result)
            lst = word_counter.most_common(10)
            print(genre, lst[0])
            df = pd.DataFrame(lst, columns=['Word', 'Count'])
            df.plot.bar(x='Word', y='Count')
            plt.show()

        graph_for_number_of_words(number_of_words, 'Genre')

    csv_file.close()


# create csv file
def generate_bad_words_score_by_decades():

    with open('result-score_by_decades.csv', mode='w') as csv_file:

        decade = 1930
        while decade != 2020:
            score = 0

            while score != 11:
                path = os.path.join('score_by_decades', str(decade), 'score_' + str(score))
                if not os.path.exists(path):
                    score += 1
                    continue
                result = extract_bad_words(path)
                fieldnames = ['decade', 'score', 'bad word', 'number of appearance']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in result:
                    writer.writerow({'decade': decade, 'score': score, 'bad word': res[0], 'number of appearance': res[1]})

                score += 1

            decade += 10

    csv_file.close()


def generate_bad_words_genre_by_decades():
    with open('result-genre_by_decades.csv', mode='w') as csv_file:

        decade = 1930
        while decade != 2020:

            for genre in ['Western', 'War', 'TV', 'Thriller', 'Science', 'Romance', 'Mystery',
                          'Music', 'Horror', 'History', 'Foreign', 'Fantasy', 'Family', 'Drama',
                          'Documentary', 'Crime', 'Comedy', 'Animation', 'Adventure', 'Action']:

                path = os.path.join('genres_by_decades', str(decade), genre)
                if not os.path.exists(path):
                    continue
                result = extract_bad_words(path)

                fieldnames = ['decade', 'genre', 'bad word', 'number of appearance']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in result:
                    writer.writerow({'decade': decade, 'genre': genre, 'bad word': res[0],
                                     'number of appearance': res[1]})

            decade += 10

    csv_file.close()


def score_by_decades_graphs():
    number_of_words = dict()

    decade = 1930
    while decade != 2020:
        score = 0

        while score != 11:
            path = os.path.join('score_by_decades', str(decade), 'score_' + str(score))
            if not os.path.exists(path):
                score += 1
                continue
            result = extract_bad_words(path)
            all_words = find_num_all_word(os.path.join('score', 'score_' + str(score)))
            number_of_words[str(score)] = (len(result.keys()) / all_words) * 100

            score += 1

        word_counter = collections.Counter(number_of_words)
        lst = word_counter.most_common(10)
        df = pd.DataFrame(lst, columns=["Score", 'Number'])
        df.plot.bar(x="Score", y='Number',
                    title="percentage of offensive words in each Score for decade {}".format(decade))
        plt.show()

        decade += 10


"""
 Analysing the word "GOD"
"""


def appearance_god_dictionary(year):
    with open(year, 'r',  encoding="utf-8") as f:
        text = f.read()

    good = ["god bless you", "god bless", "love of god", "god love",
            "god forbid", "god know", "god only knows", "god willing", "god's gift",
            "god's honest truth", "good god", "thank god", "true as god", "trust in god",
            "god grant", "god help", "asked god", "ask god", "god promise", "god forgive"]

    bad = ["god hate", "god damn"]

    dic = dict()

    for sentence in text.split("\n"):
        for g in good:
            if g in sentence:

                if g in dic.keys():
                    dic[g] += 1
                else:
                    dic[g] = 1
    return dic


def appearance_per_decade():
    decade = 1930

    while decade != 2020:

        result = appearance_god_dictionary(os.path.join('decades', str(decade)))
        print(decade, result)
        decade += 10


def helper(year):
    with open(year, 'r', encoding="utf-8") as f:
        text = f.read()

        for sentence in text.split("\n"):
            for word in sentence.split(" "):
                if word.lower() == 'god':
                    print(sentence)
                    continue


def extract_god_sentences(year):
    good = ["god bless you", "god bless", "love of god", "god love",
            "god forbid", "god know", "god only knows", "god willing", "god's gift",
            "god's honest truth", "good god", "thank god", "true as god", "trust in god",
            "god grant", "god help", "asked god", "ask god", "god promise", "god forgive"]

    bad = ["god hate", "god damn"]

    counter = 0

    with open(year, 'r',  encoding="utf-8") as f:
        text = f.read()

        for sentence in text.split("\n"):
            for g in good:
                if g in sentence.lower():
                    counter += 1
                    continue
    return counter


def generate_god_sentences_by_decades():
    decade = 1930
    result = 0

    while decade != 2020:
        result += extract_god_sentences(os.path.join('decades', str(decade)))

        decade += 10
    print(result)


def find_specific_appearance(good_word, year):
    with open(year, 'r',  encoding="utf-8") as f:
        text = f.read()

    counter = 0

    for sentence in text.split("\n"):
        if good_word in sentence.lower():
            # print(good_word)
            counter += 1
    return counter


def find_differential_in_use_god_word():
    number_of_words = dict()

    good = ["god bless you", "god bless", "love of god", "god love",
            "god forbid", "god know", "god only knows", "god willing", "god's gift",
            "god's honest truth", "good god", "thank god", "true as god", "trust in god",
            "god grant", "god help", "asked god", "ask god", "god promise", "god forgive"]

    bad = ["god hate", "god damn"]

    min_arr = []
    max_arr = []
    decades_max = []
    decades_min = []
    for g in bad:

        temp = {}
        decade = 1930

        while decade != 2020:
            # num_words = extract_god_sentences(os.path.join('decades', str(decade)))
            num_words = find_num_all_word(os.path.join('decades', str(decade)))
            # print(g, decade, "all", num_words)

            num_g_appearance = find_specific_appearance(g, os.path.join('decades', str(decade)))
            perc = (num_g_appearance/num_words) * 100
            # print(g, perc)
            temp[decade] = (perc)
            # print(temp)

            decade += 10
        # print(temp)
        min_perc = min(temp.values())
        max_perc = max(temp.values())
        key_min = min(temp.keys(), key=(lambda k: temp[k]))
        key_max = max(temp.keys(), key=(lambda k: temp[k]))
        min_arr.append(min_perc)
        decades_min.append(key_min)
        max_arr.append(max_perc)
        decades_max.append(key_max)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bad,
        y=min_arr,
        text=decades_min,
        textposition='outside',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=bad,
        y=max_arr,
        text=decades_max,
        textposition='outside',

        marker_color='indianred'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis=dict(
        title='percentage',
    ), )
    fig.show()


def graph_for_large_differences():
    f = open("bad-words.txt", "r")
    bad_words = f.read()
    bad_words = bad_words.split("\n")
    del bad_words[0]

    result_for_all_decades = find_result_for_all_decades()

    max_values = {}
    min_values = {}

    for word in bad_words:
        for decade in result_for_all_decades:
            num_of_bad_words_in_a_decade = sum(result_for_all_decades[decade].values())
            if num_of_bad_words_in_a_decade == 0:
                num_of_bad_words_in_a_decade = 1

            if word in result_for_all_decades[decade]:
                if word in max_values:
                    if result_for_all_decades[decade][word] > max_values[word]['percent']:
                        max_values[word] = {'decade': decade, 'percent': (int(result_for_all_decades[decade][word])
                                                                          / num_of_bad_words_in_a_decade)}
                else:
                    max_values[word] = {'decade': decade, 'percent': (int(result_for_all_decades[decade][word])
                                                                      / num_of_bad_words_in_a_decade)}
                if word in min_values:

                    if result_for_all_decades[decade][word] <= min_values[word]['percent']:
                        min_values[word] = {'decade': decade, 'percent': (
                                    int(result_for_all_decades[decade][word]) / num_of_bad_words_in_a_decade)}
                else:
                    min_values[word] = {'decade': decade, 'percent': (
                                int(result_for_all_decades[decade][word]) / num_of_bad_words_in_a_decade)}

    y_max = []
    y_min = []
    max_decades = []
    min_decades = []
    words_in_both = []

    for word in bad_words:
        if word in max_values and word in min_values and\
                max_values[word]['percent'] - min_values[word]['percent'] > 0.01:
            words_in_both.append(word)
            y_max.append(max_values[word]['percent'])
            max_decades.append(max_values[word]['decade'])
            y_min.append(min_values[word]['percent'])
            min_decades.append(min_values[word]['decade'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=words_in_both,
        y=y_min,
        text=min_decades,
        textposition='outside',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=words_in_both,
        y=y_max,
        text=max_decades,
        textposition='outside',

        marker_color='indianred'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis=dict(
        title='offensive word in a decade / all offensive words in that decade',
    ), )
    fig.show()


if __name__ == '__main__':

    generate_bad_words_by_decades()
    # generate_bad_words_by_score()
    # generate_bad_words_by_genres()
    # generate_bad_words_score_by_decades()
    # generate_bad_words_genre_by_decades()
    # generate_god_sentences_by_decades()
    # find_differential_in_use_god_word()
    # appearance_per_decade()
    # graph_for_large_differences()
    # continues_graph_for_number_of_words()
    # score_by_decades_graphs()
    # words_in_all_decades()
