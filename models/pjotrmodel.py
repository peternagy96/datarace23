import pandas as pd
from datasets.base import Dataset
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, log_loss, mean_absolute_error
from datasets.base import Dataset
from .model import XGBModel

class PjotrXGBModel(XGBModel):
    def fit(self, ds: Dataset, cv: int = 5):
        self.train_params = {
            "cv": cv,
        }
        self.dataset_params = ds.parameters
        
        X = ds.encoded_train_df.drop(columns=[ds.target])
        y = ds.encoded_train_df[ds.target]
        
        self.model = xgb.XGBClassifier(**self.model_params)
        
        mae=[]
        log_loss_stratified=[]
        f1_scores=[]
        gen = StratifiedKFold(cv, shuffle=True, random_state=42)

        for trn_idx, test_idx in gen.split(X,y):
            xtrain,xtest=X.iloc[trn_idx],X.iloc[test_idx]
            ytrain,ytest=y.iloc[trn_idx],y.iloc[test_idx]
            self.model.fit(xtrain,ytrain, verbose=False)
            preds = self.model.predict(xtest)

            mae.append(mean_absolute_error(ytest, preds))
            log_loss_stratified.append(log_loss(ytest, preds))
            f1_scores.append(f1_score(ytest, preds))
        
        mae = pd.Series(mae).mean()
        print(f"Stratified mae: {mae}")
        log_loss_stratified = pd.Series(log_loss_stratified).mean()
        print(f"Stratified log_loss: {log_loss_stratified}")
        f1_scores = pd.Series(f1_scores).mean()
        print(f"Stratified f1_scores: {f1_scores}")
        

        return log_loss_stratified
