# -*- coding: UTF-8 -*-

## Global settings and imports
import sys
sys.path.append("../../")

import os
# from tempfile import TemporaryDirectory
import papermill as pm
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR') # only show error messages

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.io.dkn_iterator import DKNTextIterator

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")


## Download and load data
data_path = os.path.abspath(os.path.join(os.getcwd(), "../../data/mind-demo-dkn"))
data_path1 = os.path.abspath(os.path.join(os.getcwd(), "../../data"))
print('data_path:', data_path)
print('data_path1:', data_path1)

yaml_file = os.path.join(data_path, r'dkn.yaml')
train_file = os.path.join(data_path, r'train_mind_demo.txt')
valid_file = os.path.join(data_path, r'valid_mind_demo.txt')
test_file = os.path.join(data_path, r'test_mind_demo.txt')
news_feature_file = os.path.join(data_path, r'doc_feature.txt')
user_history_file = os.path.join(data_path, r'user_history.txt')
wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')
contextEmb_file = os.path.join(data_path, r'TransE_context2vec_100.npy')
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path1, 'mind-demo-dkn.zip')


## Create hyper-parameters
epochs = 10
history_size = 50
batch_size = 100
hparams = prepare_hparams(yaml_file,
                          news_feature_file = news_feature_file,
                          user_history_file = user_history_file,
                          wordEmb_file=wordEmb_file,
                          entityEmb_file=entityEmb_file,
                          contextEmb_file=contextEmb_file,
                          epochs=epochs,
                          history_size=history_size,
                          batch_size=batch_size)
print(hparams)


## Train the DKN model
model = DKN(hparams, DKNTextIterator)
print(model.run_eval(valid_file))
model.fit(train_file, valid_file)


## Evaluate the DKN model
res = model.run_eval(test_file)
print(res)
pm.record("res", res)
