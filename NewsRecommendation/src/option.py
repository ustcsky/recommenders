import argparse

parser = argparse.ArgumentParser(description='Configuration')

# Hardware specifications
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')

# Data specifications
parser.add_argument('--pre', type=bool, default=False,
                    help='preprocess the dataset')
parser.add_argument('--train_dir', type=str, default='../../data/MINDlarge_train',
                    help='training set directory')
parser.add_argument('--test_dir', type=str, default='../../data/MINDlarge_dev',
                    help='testing set directory')
parser.add_argument('--data_dir', type=str, default='../../data/data1',
                    help='directory for preprocessed data')
parser.add_argument('--embedding_file', type=str, default='../../data/glove.840B.300d.txt',
                    help='glove embedding file')
parser.add_argument('--news_docs', type=str, default='../../data/data1/news_docs.tsv',
                    help='news docs file')
parser.add_argument('--save', type=str, default=None,
                    help='directory to save')

# Model specifications
parser.add_argument('--model', type=str, default='NAML',
                    help='choose model')
parser.add_argument('--n_browsed_news', type=int, default=50,
                    help='number of browsed news per user')
parser.add_argument('--n_words_title', type=int, default=30,
                    help='number of words per title')
parser.add_argument('--n_words_abstract', type=int, default=80,
                    help='number of words per abstract')
parser.add_argument('--n_words_body', type=int, default=100,
                    help='number of words per body')
parser.add_argument('--word_freq_threshold', type=int, default=10,
                    help='threshold for the frequency of word appearance')
parser.add_argument('--word_embedding_dim', type=int, default=300,
                    help='dimension of word embedding vector')
parser.add_argument('-category_embedding_dim', type=int, default=100,
                    help='dimension of category embedding vector')
parser.add_argument('--query_vector_dim', type=int, default=200,
                    help='dimension of the query vector in attention')
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='ratio of dropout')
parser.add_argument('--n_filters', type=int, default=400,
                    help='number of filters in CNN')
parser.add_argument('--window_size', type=int, default=3,
                    help='size of filter in CNN')
parser.add_argument('-n_categories', type=int, default=296,
                    help='number of categories and subcategories')
parser.add_argument('-n_words', type=int, default=21702,
                    help='number of words')
parser.add_argument('-n_hot_clicks', type=int, default=60639,
                    help='number of hots')
parser.add_argument('-n_hot_displays', type=int, default=331547,
                    help='number of hots')
parser.add_argument('-n_burst_clicks', type=int, default=151596,
                    help='number of hots')
parser.add_argument('-n_burst_displays', type=int, default=208110,
                    help='number of hots')

# 60638 331546 151595 208109
# 41247 243550 103117 138015

# Training specifications
parser.add_argument('--train', type=bool, default=True,
                    help='train or test only')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128,
                    help='size of each batch')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--neg_sample', type=int, default=4,
                    help='number of negative samples in negative sampling')
parser.add_argument('--load', type=str, default=None,
                    help='load pretrained model')

# NRMS specifications
parser.add_argument('--n_heads', type=int, default=15,
                    help='heads in multi-head self-attention')

# TANR specifications
parser.add_argument('--topic_weight', type=float, default=0.2,
                    help='lambda in the loss function of TANR')

args = parser.parse_args()