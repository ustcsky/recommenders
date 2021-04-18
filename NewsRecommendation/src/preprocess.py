import pandas as pd
import numpy as np
import os
import random
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def process_behaviors(args):
    random.seed(args.seed)
    if args.train:
        print('Preprocessing behaviors in training set...')
        train_set = os.path.join(args.train_dir, 'behaviors_yq_50_80.tsv')
        train_behaviors = pd.read_table(train_set, header=None, usecols=[0, 1, 2, 3, 4], na_filter=False,
                                        names=['impression_id','user_id','time','history','impressions'])
        train_behaviors.history = train_behaviors.history.astype(str).str.split(' ')
        train_behaviors.impressions = train_behaviors.impressions.astype(str).str.split(' ')

        # encode users
        user_dict = {}
        for row in train_behaviors.itertuples():
            if row.user_id not in user_dict:
                user_dict[row.user_id] = len(user_dict) + 1
        pd.DataFrame(user_dict.items(), columns=['user_id', 'index']).to_csv(os.path.join(args.data_dir, 'user_dict.csv'), sep='\t', index=False)
        for row in train_behaviors.itertuples():
            train_behaviors.at[row.Index, 'user_id'] = user_dict[row.user_id]
        print('\ttotal_user in training set:', len(user_dict))

        # prepare training pairs
        for row in train_behaviors.itertuples():
            tmp = row.history
            random.shuffle(row.history)
            train_behaviors.at[row.Index, 'history'] = ' '.join(tmp)
            pos = [i.split('-')[0] for i in row.impressions if i.endswith('1')]
            neg = [i.split('-')[0] for i in row.impressions if i.endswith('0')]
            random.shuffle(pos)
            random.shuffle(neg)
            pos = iter(pos)
            neg = iter(neg)
            training_pairs = []
            try:
                while True:
                    tmp = [next(pos)]
                    for i in range(args.neg_sample):
                        tmp.append(next(neg))
                    training_pairs.append(tmp)
            except StopIteration:
                pass
            train_behaviors.at[row.Index, 'impressions'] = training_pairs
        train_behaviors = train_behaviors.explode('impressions').dropna(subset=['impressions']).reset_index(drop=True)
        train_behaviors['impressions'] = train_behaviors['impressions'].apply(lambda x: ' '.join(x))
        train_behaviors.to_csv(os.path.join(args.data_dir, 'train_behaviors.csv'), sep='\t', index=False, columns=['user_id', 'history', 'impressions'])
        print('Processed behaviors data for training saved as ' + os.path.join(args.data_dir, 'train_behaviors.csv'))
        print('Finish behaviors preprocessing for training')

    print('Preprocessing behaviors in testing set...')
    test_set = os.path.join(args.test_dir, 'behaviors_yq_50_80.tsv')
    test_behaviors = pd.read_table(test_set, header=None, usecols=[0, 1, 2, 3, 4], na_filter=False,
                                    names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    test_behaviors.history = test_behaviors.history.astype(str).str.split(' ')
    test_behaviors.impressions = test_behaviors.impressions.astype(str).str.split(' ')

    for row in test_behaviors.itertuples():
        tmp = row.history
        # print('tmp[0]:', tmp[0])
        random.shuffle(row.history)
        test_behaviors.at[row.Index, 'history'] = ' '.join(tmp)
        tmp = row.impressions
        random.shuffle(tmp)
        # print(tmp)
        y_true = [i.split('-')[1] for i in tmp]
        candidate_news = [i.split('-')[0] for i in tmp]
        test_behaviors.at[row.Index, 'impressions'] = ' '.join(candidate_news)
        test_behaviors.at[row.Index, 'y_true'] = ' '.join(y_true)
    test_behaviors.to_csv(os.path.join(args.data_dir, 'test_behaviors.csv'), sep='\t', index=False,
                           columns=['impression_id', 'history', 'impressions', 'y_true'])
    print('Processed behaviors data for testing saved as ' + os.path.join(args.data_dir, 'test_behaviors.csv'))
    print('Finish behaviors preprocessing for testing')

def process_news(args):
    random.seed(args.seed)

    if os.path.exists(args.news_docs):
        news_docs = pd.read_table(args.news_docs, header=None, usecols=[0, 6], names=['nid', 'body'], index_col=0, na_filter=False)

    if args.train:
        print('Preprocessing news in training set...')
        # train_set = os.path.join(args.train_dir, 'news.tsv')
        train_set = os.path.join(args.train_dir, 'news_hot_burst.tsv')
        train_news = pd.read_table(train_set, header=None, usecols=[0, 1, 2, 3, 4, 5,6,7,8,9], na_filter=False,
                                   names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'body','hot_click','hot_display','burst_click','burst_display'])

        # encode category, subcategory and word
        category_dict, word_freq_dict, word_dict = {}, {}, {}
        with tqdm(total=len(train_news), desc='Processing training news') as p:
            for row in train_news.itertuples():
                if os.path.exists(args.news_docs):
                    nid = row.body.split('/')[-1].split('.')[-2]
                    news_body = news_docs.loc[nid, 'body']
                else:
                    news_body = ''
                if news_body == '-':
                    news_body = ''
                train_news.at[row.Index, 'body'] = news_body

                if row.category not in category_dict:
                    category_dict[row.category] = len(category_dict) + 1
                if row.subcategory not in category_dict:
                    category_dict[row.subcategory] = len(category_dict) + 1
                for word in word_tokenize(row.title.lower()):
                    word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                for word in word_tokenize(row.abstract.lower()):
                    word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                for word in word_tokenize(train_news.at[row.Index, 'body'].lower()):
                    word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                p.update(1)

        for k, v in word_freq_dict.items():
            if v >= args.word_freq_threshold:
                word_dict[k] = len(word_dict) + 1
        pd.DataFrame(category_dict.items(), columns=['category', 'index']).to_csv(os.path.join(args.data_dir, 'category_dict.csv'),
                                                                             sep='\t', index=False)
        pd.DataFrame(word_dict.items(), columns=['word', 'index']).to_csv(os.path.join(args.data_dir, 'word_dict.csv'),
                                                                          sep='\t', index=False)
        args.n_categories = len(category_dict) + 1
        args.n_words = len(word_dict) + 1
        print('\ttotal_category:', len(category_dict))
        print('\ttotal_word:', len(word_dict))
        print('\tRemember to set n_categories = {} and n_words = {} in option.py'.format(len(category_dict) + 1, len(word_dict) + 1))

        with tqdm(total=len(train_news), desc='Encoding training news') as p:
            for row in train_news.itertuples():
                train_news.at[row.Index, 'category'] = category_dict[row.category]
                train_news.at[row.Index, 'subcategory'] = category_dict[row.subcategory]
                title = [0 for _ in range(args.n_words_title)]
                abstract = [0 for _ in range(args.n_words_abstract)]
                body = [0 for _ in range(args.n_words_body)]
                try:
                    for i, word in enumerate(word_tokenize(row.title.lower())):
                        if word in word_dict:
                            title[i] = word_dict[word]
                except IndexError:
                    pass
                try:
                    for i, word in enumerate(word_tokenize(row.abstract.lower())):
                        if word in word_dict:
                            abstract[i] = word_dict[word]
                except IndexError:
                    pass
                try:
                    for i, word in enumerate(word_tokenize(row.body.lower())):
                        if word in word_dict:
                            body[i] = word_dict[word]
                except IndexError:
                    pass
                train_news.at[row.Index, 'title'] = title
                train_news.at[row.Index, 'abstract'] = abstract
                train_news.at[row.Index, 'body'] = body
                train_news.at[row.Index, 'hot_click'] = row.hot_click
                train_news.at[row.Index, 'hot_display'] = row.hot_display
                train_news.at[row.Index, 'burst_click'] = row.burst_click
                train_news.at[row.Index, 'burst_display'] = row.burst_display
                p.update(1)
        train_news.to_csv(os.path.join(args.data_dir, 'train_news.csv'), sep='\t', index=False,
                          columns=['news_id', 'category', 'subcategory', 'title', 'abstract', 'body','hot_click','hot_display','burst_click','burst_display'])
        print('Finish news preprocessing for training')
    
    print('Preprocessing news in testing set...')
    test_set = os.path.join(args.test_dir, 'news_hot_burst.tsv')
    test_news = pd.read_table(test_set, header=None, usecols=[0, 1, 2, 3, 4, 5,6,7,8,9], na_filter=False,
                               names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'body','hot_click','hot_display','burst_click','burst_display'])

    # encode category, subcategory and word
    category_dict = dict(pd.read_csv(os.path.join(args.data_dir, 'category_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())
    word_dict = dict(pd.read_csv(os.path.join(args.data_dir, 'word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())

    with tqdm(total=len(test_news), desc='Encoding test news') as p:
        for row in test_news.itertuples():
            if os.path.exists(args.news_docs):
                nid = row.body.split('/')[-1].split('.')[-2]
                news_body = news_docs.loc[nid, 'body']
            else:
                news_body = ''
            if news_body == '-':
                news_body = ''
            test_news.at[row.Index, 'body'] = news_body

            if row.category in category_dict:
                test_news.at[row.Index, 'category'] = category_dict[row.category]
            else:
                test_news.at[row.Index, 'category'] = 0
            if row.subcategory in category_dict:
                test_news.at[row.Index, 'subcategory'] = category_dict[row.subcategory]
            else:
                test_news.at[row.Index, 'subcategory'] = 0

            title = [0 for _ in range(args.n_words_title)]
            abstract = [0 for _ in range(args.n_words_abstract)]
            body = [0 for _ in range(args.n_words_body)]
            try:
                for i, word in enumerate(word_tokenize(row.title.lower())):
                    if word in word_dict:
                        title[i] = word_dict[word]
            except IndexError:
                pass
            try:
                for i, word in enumerate(word_tokenize(row.abstract.lower())):
                    if word in word_dict:
                        abstract[i] = word_dict[word]
            except IndexError:
                pass
            try:
                for i, word in enumerate(word_tokenize(test_news.at[row.Index, 'body'] .lower())):
                    if word in word_dict:
                        body[i] = word_dict[word]
            except IndexError:
                pass
            test_news.at[row.Index, 'title'] = title
            test_news.at[row.Index, 'abstract'] = abstract
            test_news.at[row.Index, 'body'] = body
            test_news.at[row.Index, 'hot_click'] = row.hot_click
            test_news.at[row.Index, 'hot_display'] = row.hot_display
            test_news.at[row.Index, 'burst_click'] = row.burst_click
            test_news.at[row.Index, 'burst_display'] = row.burst_display
            p.update(1)

    test_news = test_news.append([{'news_id': 0, 'category': 0, 'subcategory': 0, 'title': [0 for _ in range(args.n_words_title)],
                                   'abstract': [0 for _ in range(args.n_words_abstract)], 'body': [0 for _ in range(args.n_words_body)],
                                   'hot_click': 0,'hot_display': 0,'burst_click': 0,'burst_display': 0}],
                                   ignore_index=True)
    test_news.to_csv(os.path.join(args.data_dir, 'test_news.csv'), sep='\t', index=False,
                      columns=['news_id', 'category', 'subcategory', 'title', 'abstract', 'body','hot_click','hot_display','burst_click','burst_display'])
    print('Finish news preprocessing for testing')


def word_embedding(args):
    if not args.train:
        return
    print('Start word embedding...')
    word_dict = dict(pd.read_csv(os.path.join(args.data_dir, 'word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())
    glove_embedding = pd.read_table(args.embedding_file, sep=' ', header=None, index_col=0, quoting=3)
    embedding_result = np.random.normal(size=(len(word_dict) + 1, args.word_embedding_dim))
    word_missing = 0
    with tqdm(total=len(word_dict), desc="Generating word embedding") as p:
        for k, v in word_dict.items():
            if k in glove_embedding.index:
                embedding_result[v] = glove_embedding.loc[k].tolist()
            else:
                word_missing += 1
            p.update(1)
    np.save(os.path.join(args.data_dir, 'word_embedding.npy'), embedding_result)
    print('embedding_result.shape:', embedding_result.shape)
    print('embedding_result[0]:', embedding_result[0])
    print('\ttotal_missing_word:', word_missing)
    print('Finish word embedding')

def preprocess_data(args):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    print('Start data preprocessing...')
    process_behaviors(args)
    process_news(args)
    word_embedding(args)
    print('Finish data preprocessing...')