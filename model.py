import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

## FUNCTION DEFINITIONS

# Func to Load the training and test dataset from Google Drive
def load_data(file_path):
    return pd.read_csv(file_path)

# For Preprocessing of Currency (Converting 1 Crore+ to 10000000+ etc)
def preprocess_currency(x):
    if isinstance(x, str):
        if 'Crore+' in x:
            return float(x.replace(' Crore+', '')) * 10000000
        elif 'Lac+' in x:
            return float(x.replace(' Lac+', '')) * 100000
        elif 'Thou+' in x:
            return float(x.replace(' Thou+', '')) * 1000
        elif 'Hund+' in x:
            return float(x.replace(' Hund+', '')) * 1000
        else:
            return float(x.replace('$', '').replace(',', ''))
    return x

# Apply currency preprocessing on Total Assets and Liabilities
def preprocess_data(data):
    data['Total Assets'] = data['Total Assets'].apply(preprocess_currency)
    data['Liabilities'] = data['Liabilities'].apply(preprocess_currency)
    return data

# Ploting
def plot_unique_values(data, title):
    unique_values = data.nunique()
    plt.figure(figsize=(10, 6))
    unique_values.plot(kind='bar', color='lightcoral')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Number of Unique Values')
    plt.show()

# Ploting
def plot_bar_chart(data, x, y, title, xlabel, ylabel, rotation=45, color='skyblue'):
    plt.figure(figsize=(14, 6))
    data.plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

def preprocess_categorical(data, columns):
    return pd.get_dummies(data, columns=columns)

def scale_features(data, features):
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

# Grid Search for Random Forest Model
def train_model(model, X_train, y_train, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_model, best_params, best_score

def predict_and_save_results(model, test_data, file_path):
    predictions = model.predict(test_data)
    index_values = np.arange(len(predictions))
    predict_df = pd.DataFrame({'ID': index_values, 'Education': predictions})
    predict_df.to_csv(file_path, index=False)

## MODEL

# Load and preprocess data
train_data = load_data('drive/MyDrive/train.csv')
test_data = load_data('drive/MyDrive/test.csv')

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Plot unique values in each column
plot_unique_values(train_data, 'Number of Unique Values in Each Column')

# Drop unnecessary columns
train_data.drop(['ID','Candidate','Constituency ∇'], axis=1, inplace=True)
test_data.drop(['ID','Candidate','Constituency ∇'], axis=1, inplace=True)

# Scale numerical features
numerical_features = ['Total Assets', 'Liabilities', 'Criminal Case']
train_data = scale_features(train_data, numerical_features)
test_data = scale_features(test_data, numerical_features)

# Preprocess categorical data
train_data = preprocess_categorical(train_data, ['Party','state'])
test_data = preprocess_categorical(test_data, ['Party','state'])

# Separate target variable
X_train = train_data.drop('Education', axis=1)
y_train = train_data['Education']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Random Forest Model
rfc_classifier = RandomForestClassifier(random_state=42)
rfc_param_grid = {'n_estimators': [500, 750, 1000, 1250, 1500]}
best_rfc_model, best_rfc_params, best_rfc_score = train_model(rfc_classifier, X_train, y_train, rfc_param_grid)
print("Random Forest Best Parameters:", best_rfc_params)
print("Random Forest Best Score:", best_rfc_score)

# Validate Random Forest Model
y_val_pred_rfc = best_rfc_model.predict(X_val)
print("Random Forest Validation Report:")
print(classification_report(y_val, y_val_pred_rfc))

# Train Naive Bayes Model
bnc_classifier = BernoulliNB()
bnc_classifier.fit(X_train, y_train)

# Validate Naive Bayes Model
y_val_pred_bnc = bnc_classifier.predict(X_val)
print("Naive Bayes Validation Report:")
print(classification_report(y_val, y_val_pred_bnc))

# Predict using Random Forest Model and save results
predict_and_save_results(best_rfc_model, test_data, 'drive/MyDrive/pred_rf.csv')

# Predict using Naive Bayes Model and save results
predict_and_save_results(bnc_classifier, test_data, 'drive/MyDrive/pred_naive_bayes.csv')
