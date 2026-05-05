import pandas as pd
import numpy as np
import xgboost as xgb


class RankingModel:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.05,
            n_estimators=200,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def train(self, X_train, y_train, groups ):
        self.model.fit(X_train, y_train, group=groups)

    def rank(self, candidates):
        return self.model.predict(candidates)