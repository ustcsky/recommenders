# News Recommendation

Implementation of several news recommendation methods in Pytorch.



### Requirements

- torch
- numpy
- pandas
- nltk
- scikit-learn
- matplotlib



### Usage

- **Clone this repository**	

  ```bash
  git clone https://github.com/yflyl613/NewsRecommendation.git
  cd NewsRecommendation
  mkdir dataset
  ```

- **Prepare dataset**

  - Dowload GloVe pre-trained word embedding (https://nlp.stanford.edu/data/glove.840B.300d.zip)

  - Unzip and put it `NewsRecommendation/dataset`

  - Create a new directory `NewsRecommendation/dataset/MINDsmall`

  - Download MIND-small dataset

    - Training Set: https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
    - Validation Set: https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip

  - Unzip and put them in `NewsRecommendation/dataset/MINDsmall`

  - After those operations above, the sturcture of `NewsRecommendation/dataset` should look like:

    ```
    |--- NewsRecommendation/
    	|--- dataset/
    		|--- MINDsmall/
    			|--- MINDsmall_dev
    			|--- MINDsmall_train
    		|--- glove.840B.300d.txt
    ```

- **Start training**

  ```bash
  cd NewsRecommendation/src
  python main.py  # add command line arguments behind, see `option.py` for details
  # eg. python main.py --model NRMS
  ```

  

### Result

- **NAML**

  |   News representation   |  AUC  |  MRR  | nDCG@5 | nDCG@10 |      Configuration       |
  | :---------------------: | :---: | :---: | :----: | :-----: | :----------------------: |
  |          title          | 66.39 | 31.76 | 35.37  |  41.53  |  batch size 64, 1 epoch  |
  |          body           | 67.29 | 32.56 | 35.83  |  42.08  |  batch size 64, 2 epoch  |
  |    title + abstract     | 67.84 | 31.87 | 35.78  |  41.91  | batch size 64, 4 epochs  |
  |      title + body       | 67.88 | 32.82 | 36.36  |  42.55  |  batch size 64, 1 epoch  |
  | title + abstract + body | 67.71 | 31.93 | 35.40  |  41.81  | batch size 128, 3 epochs |

- **NRMS**

  | News representation |  AUC  |  MRR  | nDCG@5 | nDCG@10 |      Configuration       |
  | :-----------------: | :---: | :---: | :----: | :-----: | :----------------------: |
  |        title        | 65.93 | 31.40 | 34.51  |  40.83  | batch size 192, 5 epochs |

- **TANR**

  | News representation |  AUC  |  MRR  | nDCG@5 | nDCG@10 |      Configuration       |
  | :-----------------: | :---: | :---: | :----: | :-----: | :----------------------: |
  |        title        | 65.91 | 30.94 | 34.36  |  40.61  | batch size 128, 3 epochs |

### Citation

[1] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. **MIND: A Large-scale Dataset for News Recommendation.** ACL 2020.

