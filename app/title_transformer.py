from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TitleCount(BaseEstimator, TransformerMixin):


    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(lambda x: len([
          wrd for wrd in x.split() if wrd.istitle()]))
        X_tagged = X_tagged.map(lambda x: 5 if x>5 else x)
        return pd.DataFrame(X_tagged)
