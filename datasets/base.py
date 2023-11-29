import json
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset(ABC):
    def __init__(
        self,
        ohe_features: list,
        le_features: list,
        categorical_features: list,
        log_transform_features: list,
        numerical_features: list,
        target: str = "label",
    ):
        self.save_id = (
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}-{secrets.token_hex(16)[:5]}"
        )
        self.ohe_features = ohe_features
        self.le_features = le_features
        self.categorical_features = categorical_features
        self.log_transform_features = log_transform_features
        self.numerical_features = numerical_features
        self.target = target

        self.test_users = pd.read_csv("data/data_submission_example.csv")["BORROWER_ID"].unique()
        self.train_df = pd.read_csv("data/training_data.csv")
        self.basic_transformations()
        self.train_df = self.do_feature_engineering(self.train_df)
        self.encoded_train_df = self.encode(self.train_df, label=True, keep_ids=True)
        self.encoded_test_df = self.create_test_df(self.encoded_train_df)
        self.keep_only_labelled_credits()
        self.encoded_train_df = self.encoded_train_df[~self.encoded_train_df["BORROWER_ID"].isin(self.test_users)].drop(columns=["BORROWER_ID"])


    @property
    def parameters(self) -> dict:
        return {
            "save_id": self.save_id,
            "ohe_features": self.ohe_features,
            "le_features": self.le_features,
            "categorical_features": self.categorical_features,
            "log_transform_features": self.log_transform_features,
            "numerical_features": self.numerical_features,
            "target": self.target,
        }

    def save(self) -> str:
        # create folder in models folder with the model id using Path
        if not Path(f"data/datasets/{self.__class__.__name__}/{self.save_id}").exists():
            Path(f"data/datasets/{self.__class__.__name__}/{self.save_id}").mkdir(
                parents=True, exist_ok=True
            )
        self.save_parameters()
        self.train_df.to_csv(
            f"data/datasets/{self.__class__.__name__}/{self.save_id}/train_df.csv",
            index=False,
        )
        self.test_df.to_csv(
            f"data/datasets/{self.__class__.__name__}/{self.save_id}/test_df.csv",
            index=False,
        )
        self.encoded_train_df.to_csv(
            f"data/datasets/{self.__class__.__name__}/{self.save_id}/encoded_train_df.csv",
            index=False,
        )
        self.encoded_test_df.to_csv(
            f"data/datasets/{self.__class__.__name__}/{self.save_id}/encoded_test_df.csv",
            index=False,
        )

        return self.save_id

    def save_parameters(self):
        # save parameters to a json file
        with open(
            f"data/datasets/{self.__class__.__name__}/{self.save_id}/parameters.json",
            "w",
        ) as f:
            json.dump(self.parameters, f, indent=4)

    @classmethod
    def load(cls, save_id: str):
        # check if model_id exists
        if not Path(f"data/datasets/{cls.__name__}/{save_id}").exists():
            raise ValueError(f"Dataset with id {save_id} does not exist.")

        # load parameters from json file
        obj = cls.__new__(cls)
        with open(f"data/datasets/{cls.__name__}/{save_id}/parameters.json", "r") as f:
            parameters = json.load(f)
        obj.ohe_features = parameters["ohe_features"]
        obj.le_features = parameters["le_features"]
        obj.categorical_features = parameters["categorical_features"]
        obj.log_transform_features = parameters["log_transform_features"]
        obj.numerical_features = parameters["numerical_features"]
        obj.target = parameters["target"]
        obj.save_id = save_id

        obj.train_df = pd.read_csv(
            f"data/datasets/{cls.__name__}/{save_id}/train_df.csv"
        )
        obj.encoded_train_df = pd.read_csv(
            f"data/datasets/{cls.__name__}/{save_id}/encoded_train_df.csv"
        )
        obj.test_df = pd.read_csv(f"data/datasets/{cls.__name__}/{save_id}/test_df.csv")
        obj.encoded_test_df = pd.read_csv(
            f"data/datasets/{cls.__name__}/{save_id}/encoded_test_df.csv"
        )

        return obj

    def basic_transformations(self):
        """Implement all basic transformations here."""
        self.convert_to_datetime()
        self.create_label()

    def convert_to_datetime(self) -> pd.DataFrame:
        for feature in [
            "CONTRACT_DATE_OF_LOAN_AGREEMENT",
            "CONTRACT_MATURITY_DATE",
            "TARGET_EVENT_DAY",
        ]:
            self.train_df[feature] = pd.to_datetime(
                self.train_df[feature], origin="julian", unit="D"
            )

    def keep_only_labelled_credits(self):
        """Only keep credits with the following specs:
        - target is K -> 1
        - target is E -> 0
        - is in the first year of the dataset -> 0
        """
        test_users = self.encoded_test_df["BORROWER_ID"].unique()
        self.train_df = self.train_df[~self.train_df["BORROWER_ID"].isin(test_users)]

    def create_label(self):
        """Create a label column from TARGET_EVENT."""
        self.train_df["label"] = self.train_df["TARGET_EVENT"].apply(
            lambda x: 1 if x == "K" else 0
        )

    @abstractmethod
    def do_feature_engineering(self, df):
        """Implement all feature engineering steps here."""
        raise NotImplementedError

    def encode(self, df: pd.DataFrame, keep_ids: bool = False, label: bool = True) -> pd.DataFrame:
        """Implements all feature encodings"""

        borrower_id = ["BORROWER_ID"] if keep_ids else []

        columns = (
            borrower_id
            + self.ohe_features
            + self.le_features
            + self.categorical_features
            + self.log_transform_features
            + self.numerical_features
        )
        if label:
            columns += [self.target]
        df = df[columns]

        for feature in self.ohe_features:
            df = self.one_hot_encode(df, feature)

        for feature in self.le_features:
            df = self.label_encode(df, feature)

        for feature in self.categorical_features:
            df[feature] = df[feature].astype("category")

        for feature in self.log_transform_features:
            df = self.log_transform(df, feature)

        for feature in self.numerical_features:
            df.loc[:, feature] = df[feature].astype(float)

        return df

    def one_hot_encode(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """One hot encode categorical features."""
        df = pd.get_dummies(df, columns=[feature], prefix=[feature])
        return df

    def label_encode(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Label encode categorical features."""
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        return df

    def log_transform(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Log transform numerical features."""
        df[feature] = df[feature].apply(lambda x: np.where(x > 0, np.log1p(x), 0))
        return df

    def create_test_df(self, df):
        """Create a test dataframe with needed, unique borrower IDs and same features as the train_df."""
        self.test_df = self.train_df[self.train_df["BORROWER_ID"].isin(self.test_users)].copy()
        return df[df["BORROWER_ID"].isin(self.test_users)].copy().drop(columns=["label"])
