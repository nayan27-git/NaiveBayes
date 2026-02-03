import yaml
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, log_loss, confusion_matrix

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.outlook_mode = None
        self.temp_mode = None
        self.humidity_mode = None
        self.wind_mode = None

    def _clean_missing(self, df):
        df = df.copy()
        df['Outlook'] = df['Outlook'].fillna(self.outlook_mode)
        df['Temperature'] = df['Temperature'].fillna(self.temp_mode)
        df['Humidity'] = df['Humidity'].fillna(self.humidity_mode)
        df['Wind'] = df['Wind'].fillna(self.wind_mode)
        return df

    def fit(self, X, y=None):
        X = X.copy()
        self.outlook_mode = X['Outlook'].mode()[0]
        self.temp_mode = X['Temperature'].mode()[0]
        self.humidity_mode = X['Humidity'].mode()[0]
        self.wind_mode = X['Wind'].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(['Day'],axis=1)
        X = self._clean_missing(X)
        return X
    
class ModelManager():
    def __init__(self, config):
        # Accessing nested values from the dictionary
        self.feat_cols = config['features']['categorical_columns']
        self.model_params = config['model_params']
        # Initialize model with params from config
        self.model = MultinomialNB(**self.model_params)
        self.pipeline = None
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), self.feat_cols)
            ],
            remainder='passthrough'
        )
        print(f"------Model Initialized with {len(self.feat_cols)} features------")

    def _build_pipeline(self):
        self.pipeline = Pipeline(steps=[
                ('cleaner', DataTransformer()),
                ('column transformer', self.column_transformer),
                ('model', self.model)
        ])
        return self

    def train(self, X_train, y_train):
        if self.pipeline is None:
            self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label='Yes'),
            "Recall": recall_score(y_test, y_pred,pos_label='Yes'),
            "F1 Score": f1_score(y_test, y_pred,pos_label='Yes'),
            "Log Loss": log_loss(y_test, y_pred_proba),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
        return metrics
        
