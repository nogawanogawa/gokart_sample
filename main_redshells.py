import glob

import chardet
import luigi
import luigi.format
import pickle
import subprocess
import pandas as pd
import numpy as np
import os
from model.scdv import SCDV
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

from sklearn.metrics import classification_report, accuracy_score, make_scorer

import gokart
import redshells.data
import redshells.train


def tokenize(file_path):
    p = subprocess.run(['mecab', '-Owakati', file_path],
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=False)
    try:
        lines = p.stdout.decode(chardet.detect(p.stdout)["encoding"])
        return lines.split()
    except:
        return None


class PrepareLivedoorNewsData(gokart.TaskOnKart):
    task_namespace = 'examples'
    # text_data_file_path = luigi.Parameter()  # type: str

    def run(self):
        categories = [
            'dokujo-tsushin', 'it-life-hack', 'kaden-channel', 'livedoor-homme', 'movie-enter', 'peachy', 'smax',
            'sports-watch', 'topic-news'
        ]

        data = pd.DataFrame([(c, tokenize(path)) for c in categories for path in glob.glob(f'data/text/{c}/*.txt')],
                            columns=['category', 'text'])
        data.dropna(inplace=True)
        self.dump(data)


class PrepareCorpus(gokart.TaskOnKart):
    task_namespace = 'examples'
    corpus_file_path = luigi.Parameter(default='corpus.pkl')  # type: str
    
    def requires(self):
        return PrepareLivedoorNewsData()

    def output(self):
        return self.make_target(self.corpus_file_path, use_unique_id=False)

    def run(self):
        data = self.load()
        corpus = data["text"]
        self.dump(corpus.values.tolist())

class TrainSCDV(gokart.TaskOnKart):
    
    task_namespace = 'examples'
    text_data_file_path = "corpus.pkl"

    def requires(self):
        text_data = PrepareCorpus(corpus_file_path="corpus.pkl")
        #text_data = redshells.data.LoadDataOfTask(data_task=data_task, target_name='train')
        dictionary = redshells.train.TrainDictionary(tokenized_text_data_task=text_data)
        fasttext = redshells.train.TrainFastText(tokenized_text_data_task=text_data)
        scdv = redshells.train.TrainSCDV(
            tokenized_text_data_task=text_data, dictionary_task=dictionary, word2vec_task=fasttext)
        return scdv

    def output(self):
        return self.input()

class PrepareClassificationData(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        return dict(data=PrepareLivedoorNewsData(), model=TrainSCDV())

    def run(self):
        data = self.load("data")
        model = self.load('model')  # type: SCDV

        data['embedding'] = list(model.infer_vector(
            data['text'].tolist(), l2_normalize=True))
        data = data[['category', 'embedding']].copy()
        data['category'] = data['category'].astype('category')
        data['category_code'] = data['category'].cat.codes

        self.dump(data)


class TrainClassificationModel(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        return PrepareClassificationData()

    def run(self):
        data = self.load()
        data = data.sample(frac=1).reset_index(drop=True)
        x = data['embedding'].tolist()
        y = data['category_code'].tolist()
        model = lgb.LGBMClassifier(objective="multiclass")
        scores = []
        def _scoring(y_true, y_pred):
            scores.append(classification_report(y_true, y_pred))
            return accuracy_score(y_true, y_pred)
        cross_val_score(model, x, y, cv=3, scoring=make_scorer(_scoring))
#        dump(self.output()['scores'], scores)
        model.fit(x, y)
        out = {"model" : model, "scores": scores}
        self.dump(out)


class ReportClassificationResults(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        return TrainClassificationModel()

    def output(self):
        return  self.make_target('output/results.txt')

    def run(self):
        score_texts = self.load()["scores"]
        scores = np.array([self._extract_average(text)
                           for text in score_texts])
        averages = dict(
            zip(['precision', 'recall', 'f1-score', 'support'], np.average(scores, axis=0)))

        self.dump(averages)

    @staticmethod
    def _extract_average(score_text: str):
        # return 'precision', 'recall', 'f1-score', 'support'
        return [float(x) for x in score_text.split()[-4:]]

if __name__ == '__main__':

    gokart.run([
        'examples.ReportClassificationResults', '--local-scheduler', 
    ])
