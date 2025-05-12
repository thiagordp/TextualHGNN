import json
import os
import time
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import warnings

from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

# Re-run the model training and evaluation
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from nltk.corpus import stopwords


# Suppress ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Text Preprocessing
import re

from sklearn.preprocessing import LabelEncoder
import logging

# Define the path to the dataset
# DATASET = "Imprisonment-IT"
DATASET="IMDB"
dataset_path = f"data/datasets/{DATASET}"
LANG = "english"

# Define the splits
splits = ['train', 'validation', 'test']


def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time into 'HH:MM:SS.mmm' format.

    Args:
        start_time (float): The start time as returned by `time.time()`.

    Returns:
        str: The formatted time elapsed.
    """
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"


# Setup Logging
def setup_logging(log_folder='logs', log_file='diffpool_training.log'):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

setup_logging(log_file="shallow_training.log")

logging.info(f"Using dataset: {DATASET}")


# Function to load text data
def load_data(dataset_path, splits):
    data = {'split': [], 'text': [], 'label': []}

    for split in splits:
        start_time = time.time()
        split_path = os.path.join(dataset_path, split, 'raw')

        # Iterate over each class folder
        for label in os.listdir(split_path):
            class_path = os.path.join(split_path, label)

            # Read all .txt files in the class folder
            for file_name in os.listdir(class_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(class_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        data['split'].append(split)
                        data['text'].append(text)
                        data['label'].append(label)

        logging.info(f"Loading {split} set took {format_time_elapsed(start_time)}")

    return pd.DataFrame(data)


start_time = time.time()
# Load the data into a DataFrame
df = load_data(dataset_path, splits)
logging.info(f"Loading data took {format_time_elapsed(start_time)}")

logging.info(f"Data sample: \n{df.sample(n=10)}")


def preprocess_text(text: str) -> str:
    """
    Preprocesses input text by:
    1. Lowercasing the text.
    2. Removing punctuation.
    3. Replacing numbers with 'number'.
    4. Removing extra whitespace.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and preprocessed text.
    """
    # Lowercasing
    text = text.lower()

    # Remove punctuation using translation table for performance
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Replace numbers with 'number' using str.replace for efficiency
    text = re.sub(r'\d+', ' number ', text)

    tokens = text.split()

    # Remove stopwords if enabled    
    tokens = [word for word in tokens if word not in set(stopwords.words(LANG))]

    # Remove extra whitespace
    text = ' '.join(tokens)

    return text


# Apply preprocessing
start_time = time.time()
df['text'] = df['text'].apply(preprocess_text)
logging.info(f"Preprocessing {format_time_elapsed(start_time)}")


# Separate data by split
train_data = df[df['split'] == 'train']
validation_data = df[df['split'] == 'validation']
test_data = df[df['split'] == 'test']


# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))


# Fit on training data and transform all splits
X_train = tfidf_vectorizer.fit_transform(train_data['text'])
X_val = tfidf_vectorizer.transform(validation_data['text'])
X_test = tfidf_vectorizer.transform(test_data['text'])


# Initialize the Label Encoder
label_encoder = LabelEncoder()


# Fit and transform the labels for each split
y_train = label_encoder.fit_transform(train_data['label'])
y_val = label_encoder.transform(validation_data['label'])
y_test = label_encoder.transform(test_data['label'])

logging.info(f"Classes found by LabelEncoder: {label_encoder.classes_}")



# Define models and their hyperparameters for grid search
models = {
    # 'Logistic Regression': (LogisticRegression(max_iter=1000), {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    #     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    #     'max_iter': [100, 200, 500]
    # }),
    #
    # 'Support Vector Machine': (SVC(max_iter=1000), {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #     'gamma': ['scale', 'auto'],
    #     'degree': [2, 3, 4, 5]
    # }),
    #
    # 'Random Forest': (RandomForestClassifier(), {
    #     'n_estimators': [50, 100, 200, 500],
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }),
    #
    # 'Gradient Boosting': (GradientBoostingClassifier(), {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
    #     'max_depth': [3, 5, 7, 10],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'subsample': [0.5, 0.7, 1.0]
    # }),
    #
    # 'K-Nearest Neighbors': (KNeighborsClassifier(), {
    #     'n_neighbors': [3, 5, 7, 9, 11],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan', 'minkowski']
    # }),
    #
    # 'Naive Bayes': (MultinomialNB(), {
    #     'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    #     'fit_prior': [True, False]
    # }),
    #
    # 'Decision Tree': (DecisionTreeClassifier(), {
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10, 20],
    #     'min_samples_leaf': [1, 2, 4, 10],
    #     'criterion': ['gini', 'entropy', 'log_loss']
    # }),

    # 'AdaBoost': (AdaBoostClassifier(), {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]
    # }),

    # 'Extra Trees': (ExtraTreesClassifier(), {
    #     'n_estimators': [50, 100, 200, 500],
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }),
    # 'Neural Network (MLPClassifier)': (MLPClassifier(), {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'alpha': [0.0001, 0.001, 0.01],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     'max_iter': [200, 500, 1000],
    #     'batch_size': [32, 64, 128],
    #     'early_stopping': [True, False]
    # }),
    'XGBoost': (XGBClassifier(eval_metric='logloss'), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
        'max_depth': [3, 5, 7, 10],
        # 'min_child_weight': [1, 3, 5],
        # 'gamma': [0, 0.1, 0.2, 0.3],
        # 'subsample': [0.5, 0.7, 1.0],
        # 'colsample_bytree': [0.5, 0.7, 1.0]
    })
}


# Setup cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store classification reports
classification_reports = {}

# Iterate over models and apply GridSearchCV
for model_name, (model, params) in models.items():
    logging.info("\n" + "=" * 60 + "\n")
    logging.info(f"Training {model_name}...")
    
    start_time = time.time()
    grid_search = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Trained model '{model_name}' in {format_time_elapsed(start_time)}")

    # Get the best model
    best_model = grid_search.best_estimator_

    logging.info(f"Best model: {best_model}")
    logging.info(f"Parameters: {grid_search.best_params_}")

    start_time = time.time()
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    logging.info(f"Predicting on test set the model '{model_name}' in {format_time_elapsed(start_time)}")

    # Generate and store the classification report
    report = classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
    classification_reports[model_name] = report

    # Display the report
    logging.info(f"Classification Report for {model_name}:")
    logging.info(f"\n{json.dumps(report, indent=3)}")
