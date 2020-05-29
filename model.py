from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self._features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._features]


class NumericalTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        return X


class CategoricalTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


def build_model():

    numerical_variables = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                           'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3',
                           'Family_Hist_4', 'Family_Hist_5']
    categorical_variables = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                             'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
                             'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
                             'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
                             'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
                             'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3',
                             'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
                             'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
                             'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_18',
                             'Medical_History_17', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
                             'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
                             'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
                             'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
                             'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
                             'Medical_History_40', 'Medical_History_41']
    selected_variables = numerical_variables + categorical_variables

    # Numerical Pipeline
    num_var = [var for var in selected_variables if var in numerical_variables]
    steps = [('num_selector', FeatureSelector(num_var)),
             ('num_transfomer', NumericalTransformer()),
             ('impute', SimpleImputer())]
    numerical_pipeline = Pipeline(steps=steps)

    # Categorical Pipeline
    cat_var = [var for var in selected_variables if var in categorical_variables]
    steps = [('cat_selector', FeatureSelector(cat_var)),
             ('cat_transformer', CategoricalTransformer()),
             ('ohe', OneHotEncoder(handle_unknown='ignore'))]
    categorical_pipeline = Pipeline(steps=steps)

    # Feature Union
    transformer_list = [('num_pipe', numerical_pipeline), ('cat_pipe', categorical_pipeline)]
    preprocessor = FeatureUnion(transformer_list=transformer_list)

    model = RandomForestClassifier(n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
