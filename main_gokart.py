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


class TrainSCDV(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        return PrepareLivedoorNewsData()

    def run(self):
        data = self.load()
        data = data.sample(frac=1).reset_index(drop=True)

        documents = data['text'].tolist()
        embedding_size = 200
        cluster_size = 60
        sparsity_percentage = 0.04
        word2vec_parameters = dict()
        gaussian_mixture_parameters = dict()
        dictionary_filter_parameters = dict()

        model = SCDV(
            documents=documents,
            embedding_size=embedding_size,
            cluster_size=cluster_size,
            sparsity_percentage=sparsity_percentage,
            word2vec_parameters=word2vec_parameters,
            gaussian_mixture_parameters=gaussian_mixture_parameters,
            dictionary_filter_parameters=dictionary_filter_parameters)

        self.dump(model)

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
        'examples.ReportClassificationResults', '--local-scheduler'
    ])
