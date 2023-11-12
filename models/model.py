import json
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split

from datasets.base import Dataset


class Model(ABC):
    def __init__(self, params: dict):
        self.save_id = (
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}-{secrets.token_hex(16)[:5]}"
        )
        self.model = None
        self.model_params = params
        self.dataset_params = None
        self.train_params = None
        self.train_score = None
        self.test_score = None

    @property
    def parameters(self) -> dict:
        return {
            "save_id": self.save_id,
            "model_params": self.model_params,
            "dataset_params": self.dataset_params,
            "train_params": self.train_params,
            "train_score": self.train_score,
            "test_score": self.test_score,
        }

    def save(self) -> str:
        # create folder in models folder with the save id
        if not Path(f"data/models/{self.__class__.__name__}/{self.save_id}").exists():
            Path(f"data/models/{self.__class__.__name__}/{self.save_id}").mkdir(
                parents=True, exist_ok=True
            )

        self.save_parameters()
        self.save_model()
        return self.save_id

    def save_parameters(self):
        with open(
            f"data/models/{self.__class__.__name__}/{self.save_id}/parameters.json", "w"
        ) as f:
            json.dump(self.parameters, f, indent=4)

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @classmethod
    def load(cls, save_id: str):
        # check if model_id exists
        if not Path(f"data/models/{cls.__name__}/{save_id}").exists():
            raise ValueError(f"Model with id {save_id} does not exist.")

        # load parameters from json file
        obj = cls.__new__(cls)
        with open(f"data/models/{cls.__name__}/{save_id}/parameters.json", "r") as f:
            parameters = json.load(f)
        obj.model_params = parameters["model_params"]
        obj.dataset_params = parameters["dataset_params"]
        obj.train_params = parameters["train_params"]
        obj.train_score = parameters["train_score"]
        obj.test_score = parameters["test_score"]

        obj.save_id = save_id
        obj.load_model()
        return obj

    @abstractmethod
    def fit(self, ds: Dataset, test_split: float = 0.1, cv: int = 5):
        pass

    @abstractmethod
    def predict(self, ds: Dataset):
        """Outputs the predictions in the correct format for submission"""
        pass

    @abstractmethod
    def evaluate(self, ds: Dataset):
        pass


class XGBModel(Model):
    def save_model(self):
        self.model.save_model(
            f"data/models/{self.__class__.__name__}/{self.save_id}/model.json"
        )

    def load_model(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(
            f"data/models/{self.__class__.__name__}/{self.save_id}/model.json"
        )

    def fit(self, ds: Dataset, test_split: float = 0.1, cv: int = 5):
        self.train_params = {
            "test_split": test_split,
            "cv": cv,
        }
        self.dataset_params = ds.parameters

        # fit xgb classifier in a cross validation setting
        X_train, X_test, y_train, y_test = train_test_split(
            ds.encoded_train_df.drop(columns=[ds.target]),
            ds.encoded_train_df[ds.target],
            test_size=test_split,
            random_state=42,
        )
        self.model = xgb.XGBClassifier(**self.model_params)

        self.model.fit(X_train, y_train)
        scores = cross_val_score(self.model, X_train, y_train, cv=cv)
        self.train_score = [scores.mean(), scores.std()]
        print(
            f"Cross-validation scores mean: {self.train_score[0]}, std: {self.train_score[1]}"
        )

        self.test_score = self.model.score(X_test, y_test)
        self.X_test = X_test
        print(f"Test score: {self.test_score}")

        return self.test_score

    def predict(self, ds: Dataset):
        df = ds.test_df.copy()
        df["PRED"] = self.model.predict(df.drop(columns=["BORROWER_ID"]))
        return df[["BORROWER_ID", "PRED"]]

    def evaluate(self, ds: Dataset):
        pass


class LGBModel(Model):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass


class CatBoostModel(Model):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass
