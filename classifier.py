import csv
import os
import random

import numpy as np
import pandas as pd
import sys
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble._forest import ForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def parse_subtitle_files():
    """
    For each subtitle file create a new file with only offensive words from that file.
    :return: None.
    """
    # clean csv
    csv_file_name = '/Users/shai/Documents/Huji/year_999/needle/final_project_git/analysing-subtitles/result-decade.csv'
    data = pd.read_csv(csv_file_name)

    data[data['decade'] != 'decade'].to_csv(
        "/Users/shai/Documents/Huji/year_999/needle/final_project_git/analysing-subtitles/result-decade-fix.csv")
    print('Done parsing.')


def extract_offense_words_from_txt_files():
    offense_words = set(line.strip() for line in open(
        '/Users/shai/Documents/Huji/year_999/needle/final_project_git/analysing-subtitles/bad-words.txt'))
    sub_text_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_txt/'
    sub_offense_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_offense_words_only/'
    # already_processed_sub_offense_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_offense_words_only/'
    # already_processed = os.listdir(already_processed_sub_offense_path)
    sub_text_list = os.listdir(sub_text_path)
    for sub in sub_text_list:
        offense_words_list = []
        try:
            with open(sub_text_path + sub) as sub_file:
                lines = sub_file.readlines()
                for line in lines:
                    line = line.lower()
                    words_list = re.findall(r'\w+', line)
                    for word in words_list:
                        if word in offense_words:
                            offense_words_list.append(word)
            with open(sub_offense_path + sub, 'w') as sub_offense_file:
                offense_str = ", ".join(offense_words_list)
                sub_offense_file.write(offense_str)
        except UnicodeDecodeError:
            print('UnicodeDecodeError for: ' + sub)
    print('finished')


def merge_txt_files_to_csv_with_tags():
    sub_offense_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_offense_words_only/'
    sub_file_list = os.listdir(sub_offense_path)
    sub_text_list = []
    sub_year = []
    for sub in sub_file_list:
        with open(sub_offense_path + sub, 'r') as sub_file:
            year_to_decade = list(sub.split(".")[0].split(" ")[-1])
            if len(year_to_decade) > 3:
                year_to_decade[3] = '0'
                decade = "".join(year_to_decade)
                try:
                    int(decade)
                    sub_txt = sub_file.read()
                    sub_text_list.append(sub_txt)
                    sub_year.append(decade)
                except ValueError:
                    print("ValueError: " + decade + " is not integer.")
    data = {'subtitle': sub_text_list,
            'decade': sub_year}
    df = pd.DataFrame(data, columns=['subtitle', 'decade'])
    df.to_csv(
        r'/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_offense_words_only/offense_words_to_decade_csv')


def create_dataframe_for_classifier():
    movies_csv_path = '/Users/shai/Documents/Huji/year_999/needle/final_project/movies_metadata.csv'
    subs_offense_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_offense_words_only/'
    subs_text_path = '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/subs_txt/'
    subs_list = os.listdir(subs_text_path)
    subs_offense_list = os.listdir(subs_offense_path)
    # movies dataframe
    df = pd.read_csv(movies_csv_path)
    ndf_columns = list(df.columns)
    ndf_columns.append('subtitles')
    ndf_columns.append('subtitles_offense')
    # new dataframe
    ndf = pd.DataFrame(columns=ndf_columns)

    for sub in subs_list:
        if sub == ".DS_Store":
            print("Found the damn .DS_Store file")
            continue
        movie_id = sub.split(" ")[0]
        # locate row with sub id
        row = df.loc[df['id'] == movie_id]
        with open(subs_text_path + sub, 'r') as sub_file:
            content = sub_file.read()
            row['subtitles'] = content
        with open(subs_offense_path + sub, 'r') as sub_off_file:
            off_content = sub_off_file.read()
            row['subtitles_offense'] = off_content
        dic_row = row.to_dict()
        ndf = ndf.append(dic_row, ignore_index=True)

    ndf.to_csv(
        r'/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs.csv')


def add_round_score_column():
    df = pd.read_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs.csv')
    df = df[~df['adult'].str.contains("{}")]
    round_score_list = []
    for index, row in df.iterrows():
        try:
            score = row['vote_average']
            round_score = score.split(": ")[1].replace("}", "")[0]
            round_score_list.append(round_score)
        except IndexError:
            print("wtf this is blank, not black")
            round_score_list.append("")
    df['round_score'] = round_score_list
    # df.to_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs_1.csv')


def trial_and_error_subtitles_offense_decade():
    df = pd.read_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs.csv')

    # indexNames = df[df['release_date'] == 1930].index
    # df.drop(indexNames, inplace=True)

    # create a smaller file
    # g = df.head(n = 500)
    # g.to_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs_500.csv')

    # df['subtitles_offense'].replace('', np.nan, inplace=True)
    # df.dropna(subset=['subtitles_offense'], inplace=True)

    # df = pd.read_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs_500.csv')

    # remove years 1900-1920
    # df['release_date'] = df['release_date'].apply(get_decade_from_year)
    # df = df[df.release_date != '1900']
    # df = df[df.release_date != '1910']
    # df = df[df.release_date != '1920']

    # df['subtitles_offense'] = df['subtitles_offense'].apply(get_text_from_dict)
    # df['subtitles'] = df['subtitles'].apply(get_text_from_dict)
    # df.to_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs_1.csv')

    # df = pd.read_csv('/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs_500.csv')
    # col = ['Product', 'Consumer complaint narrative']

    # df = df[~(df['release_date'] == '1930')]
    # df = df[~(df['release_date'] == '2010')]

    col = ['release_date', 'subtitles_offense']

    # df = df[col]
    df = df[col]

    # df = df[pd.notnull(df['Consumer complaint narrative'])]
    df = df[pd.notnull(df['release_date'])]

    # df.columns = ['Product', 'Consumer_complaint_narrative']
    df.columns = ['release_date', 'subtitles_offense']

    # df['category_id'] = df['Product'].factorize()[0]
    df['date_id'] = df['release_date'].factorize()[0]

    # category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
    date_id_df = df[['release_date', 'date_id']].drop_duplicates().sort_values('date_id')

    # category_to_id = dict(category_id_df.values)
    date_to_id = dict(date_id_df.values)

    # id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    # id_to_date = dict(date_to_id[['date_id', 'release_date']].values)

    df.head()

    # number of offensive words by decade. not interesting.
    # fig = plt.figure(figsize=(8, 6))
    # df.groupby('release_date').subtitles_offense.count().plot.bar(ylim=0)
    # plt.show()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                             ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.subtitles_offense).toarray()
    labels = df.date_id
    # print("features shape: each subtitle is represented by " +features.shape + " features.")
    df = df[df['release_date'].str.startswith("1") | df['release_date'].str.startswith("2")]
    df['release_date_int'] = df['release_date'].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(df['subtitles_offense'],
                                                        df['release_date_int'], random_state=2)



    # count_vect = CountVectorizer()
    # X_train_counts = tfidf.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # clf = MultinomialNB().fit(X_train_tfidf, y_train)

    clf = GradientBoostingRegressor(n_estimators=100, max_depth=2, verbose=True).fit(tfidf.fit_transform(X_train), y_train)

    ## clf = RandomForestRegressor(n_estimators=10, verbose=True).fit(tfidf.fit_transform(X_train), y_train)

    predictions = clf.predict(tfidf.transform(X_test))

    # accuracy_score(predictions, y_test)
    error = mean_absolute_error(predictions, y_test)

    dict_of_decs = {}
    dict_of_decs_x = {}


    len_elements = len(predictions)
    y_test_list = y_test.tolist()
    # build diff dictionary

    decades_list = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    dict_of_decs_a = {}
    dict_of_decs_x_a = {}
    # dict_of_decs_x[i] = []

    for i in decades_list:
        dict_of_decs[i] = []
        dict_of_decs_x[i] = []

        dict_of_decs_a[i] = []
        dict_of_decs_x_a[i] = []

    counter = 0
    for decade in decades_list:
        l = []
        la = []
        l_x = []
        d0 = int(decade/10)
        for i in range(len_elements):
            d1 = int(int(y_test_list[i])/10)
            # d2 = int(int(predictions[i])/10)
            if d1 == d0:
                l.append(predictions[i])
                la.append(y_test_list[i])
                l_x.append(counter)
                counter += 1
        dict_of_decs[decade] = l
        dict_of_decs_x[decade] = l_x
        dict_of_decs_a[decade] = la
        dict_of_decs_x_a[decade] = l_x
###

    # for i in range(len_elements):
    #     diff = abs(predictions[i] - y_test_list[i])
    #     dict_of_decs[y_test_list[i]].append(diff)

    # i = 0
    # for key in dict_of_decs:
    #     l = []
    #     for elem in dict_of_decs[key]:
    #         n = random.randint(i, i+2)
    #         l.append(n)
    #     i += 1
    #     dict_of_decs_x[key] = l

    # dict_of_sum_dect = {}
    # for key in dict_of_decs:
    #     dict_of_sum_dect[key] = (sum(dict_of_decs[key]))/(len(dict_of_decs[key]))
    # sorted_dict = {k: v for k, v in sorted(dict_of_sum_dect.items(), key=lambda item: item[1])}
    # build

    # graph_predict = (X_test, predictions)
    # graph_actual = (X_test, y_test)


    # data = (graph_predict, graph_actual)
    colors = ["tomato", "dodgerblue", "orange", "limegreen", "violet", "midnightblue", "gold", "seagreen", "maroon"]
    groups = ("1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    for (k1,v1), (k2,v2), colors, groups in zip(dict_of_decs_x.items(), dict_of_decs.items(), colors, groups):
        # plt.figure()
        x = v1
        y = v2
        # plt.plot(x, y, "red")
        ax.scatter(x, y, alpha=0.8, c=colors, edgecolors='none', s=30, label=groups)
        plt.title = "Classifier predictions VS actual"
        plt.show()

    # import seaborn as sns
    #
    # sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    # sns.stripplot(x='model_name', y='accuracy', data=cv_df,
    #               size=8, jitter=True, edgecolor="gray", linewidth=2)
    # plt.show()

    # for (k1,v1), (k2,v2), in zip(dict_of_decs_x_a.items(), dict_of_decs_a.items()):
    #     i += 1
    #     x = v1
    #     y = v2
    #     plt.plot(x, y, "black")
        # ax.scatter(x, y, alpha=0.8, c='black', edgecolors='none', s=30)



    # for data, colors, groups in zip(data, colors, groups):
    #     x, y = data


    # for data, colors, groups in zip(data, colors, groups):
    #     x, y = data
    #     ax.scatter(x, y, alpha=0.8, c=colors, edgecolors='none', s=30, label=groups)
    plt.title = "Classifier predictions VS actual"
    plt.legend(loc=2)
    plt.show()

    print(error)
    # print(clf.predict(count_vect.transform(["sex, sex, god, bastard, god, toilet, angry, angry, angry, fat, bastard, christ, dead, tits, bitch, god, hell, hole, hell, hole, shit, hell, rape, die, sexy, mad, angry, god, god, hell, rape, panties, god, panties, killed, killed, god, hell, god, god, god, mad, hell, killing, hell, hell, hell, god, crime"])))
    # print(clf.predict(count_vect.transform(["idiot, god, dead, hell, hell, hell, god, die, robber, dumb, drunk, god, american, american, american, sick, kill, mad, funeral, fuck, god, horn, god, god, blind, god, failure, failure, fairy, ass, ass, idiot, sexy, ass, balls, shithead, jesus, firing, god, god, fight, kill, lingerie, died, god, die, disease, kid, god, sex, sex, jesus, tongue, jesus, christ, horn, desire, god, slut, god, ass, ass, shit, jesus, girls, tits, horn, shoot, shit, death, killed, fat, fat, fat, fat, welfare, horseshit, christ, die, sick, burn, hell, hell, girls, liquor, sex, died, angry, ass, kill, fire, fraud, fraud, fire, hell, god, god, bullshit, god, god, angry, jesus, kill, sex, shit, horn, horn, slut, rape, stupid, angry, fire, fire, gay, sex, shit, gay, gay, gay, kill, lesbian, lesbian, gay, gay, lesbian, lesbian, pissed, god, disease, laid, death, christ, girls"])))
    # print(clf.predict(count_vect.transform(["chin, chin, fart, homo, homo, girls, hell, fire, harder, die, god, goddamn, goddamn, kill, god, jesus, die, soviet, kinky, kid, bigger, hell, fuck, damn, damn, fight, fire, fire, hell, harder, hell, soviet, american, kill, jesus, killing, stupid, stupid, attack, shoot, shoot, fire, ass, fight, fear, killed, dead, soviet, shoot, killed, blow, bomb, god, beast, sexy, die, faith, lies, attack, soviet, soviet, soviet, failed, american, attack, kill, fear, fear, blind, killing, hell, attack, ass, beast, god, god, gun, fire, dead, blow, fire, fire, fire, fire, horn, weapon, disturbed, beast, bomb, beast, god, beast, bomb, god, damn, fight, weapon, killed, enemy, fear, attack, fire, fire, beast, american, attack, jesus, hell"])))
    # print(clf.predict(count_vect.transform(["die, american, american, period, period, gay, liberal, playgirl, girls, church, fight, fight, fight, girls, black, black, girls, european, black, shoot, church, church, fat, burn, blind, cigarette, church, mad, burn, church, church, ho, ho, church, church, mad, gay, mad, cigarette, cigarette, period, period, church, period, period, period"])))
    # print(clf.predict(count_vect.transform(["buried, died, god, angry, angry, devil, girls, knife, knife, knife, death, fore, hook, bigger, servant, tongue, virgin, die, knife, fight, killed, kill, knife, hole, attack, die, chinese, chinese, chinese, devil, liquor, fear, god, whiskey, killed, harder, fear, dead, die, dead, dead, drunk, drunk, drunk, church, ho, whiskey, stupid, drunken, drunken, jackass, blind, bigger, god, failed, god, god, failed, failure, failure, god, dead, dead, burn, killing, fight, stupid, bigger, bible, fight, god, god, lies, church, god"])))

    print("How are you holding up? Because I'm a potato.")


def trial_and_error_subtitles_decade():
    df = pd.read_csv(
        '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs.csv')

    col = ['release_date', 'subtitles']

    df = df[col]

    df = df[pd.notnull(df['release_date'])]

    df.columns = ['release_date', 'subtitles']

    df['date_id'] = df['release_date'].factorize()[0]

    date_id_df = df[['release_date', 'date_id']].drop_duplicates().sort_values('date_id')

    date_to_id = dict(date_id_df.values)

    df.head()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.subtitles).toarray()
    labels = df.date_id
    # print("features shape: each subtitle is represented by " +features.shape + " features.")
    df = df[df['release_date'].str.startswith("1") | df['release_date'].str.startswith("2")]
    df['release_date_int'] = df['release_date'].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(df['subtitles'],
                                                        df['release_date_int'], random_state=2)


    # count_vect = CountVectorizer()
    # X_train_counts = tfidf.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=2, verbose=True).fit(
        tfidf.fit_transform(X_train), y_train)
    # clf = RandomForestRegressor(n_estimators=10, verbose=True).fit(tfidf.fit_transform(X_train), y_train)
    predictions = clf.predict(tfidf.transform(X_test))
    # accuracy_score(predictions, y_test)
    error = mean_absolute_error(predictions, y_test)
    print(error)

    print("I see you.")


def trial_and_error_subtitles_score():
    df = pd.read_csv(
        '/Users/shai/Documents/Huji/year_999/needle/final_project_datasets/movies_metadata_with_subs.csv')

    col = ['release_date', 'subtitles']

    df = df[col]

    df = df[pd.notnull(df['release_date'])]

    df.columns = ['release_date', 'subtitles']

    df['date_id'] = df['release_date'].factorize()[0]

    date_id_df = df[['release_date', 'date_id']].drop_duplicates().sort_values('date_id')

    date_to_id = dict(date_id_df.values)

    df.head()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.subtitles).toarray()
    labels = df.date_id
    # print("features shape: each subtitle is represented by " +features.shape + " features.")
    df = df[df['release_date'].str.startswith("1") | df['release_date'].str.startswith("2")]
    df['release_date_int'] = df['release_date'].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(df['subtitles'],
                                                        df['release_date_int'], random_state=2)
    # count_vect = CountVectorizer()
    # X_train_counts = tfidf.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=2, verbose=True).fit(
        tfidf.fit_transform(X_train), y_train)
    # clf = RandomForestRegressor(n_estimators=10, verbose=True).fit(tfidf.fit_transform(X_train), y_train)
    predictions = clf.predict(tfidf.transform(X_test))
    # accuracy_score(predictions, y_test)
    error = mean_absolute_error(predictions, y_test)
    print(error)

    print("Whale hello there.")




def get_year_from_date(date):
    try:
        year = date.split(" ")[1].replace("'", "").split("-")[0]

        return year
    except IndexError:
        print(date)


def get_decade_from_year(year):
    try:
        lst = list(str(year))
        lst[3] = '0'
        decade = "".join(lst)
        return decade
    except IndexError:
        print(year)


def get_text_from_dict(dis_dic):
    try:
        text = dis_dic.split(": ")[1].replace("}", "")
        return text
    except IndexError:
        print('Boom, empty dic')


def main(args):
    trial_and_error_subtitles_offense_decade()
    # add_round_score_column()
    # parse_subtitle_files()
    # extract_offense_words_from_txt_files()
    # create_dataframe_for_classifier()
    # trial_and_error()
    print("don't know.")


if __name__ == '__main__':
    main(sys.argv)
