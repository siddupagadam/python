# python
import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time

# Load dataset
dataset = pd.read_csv("/content/Scenario-B-merged_5s.csv")

# Feature selection
targets = ['label']
feature_names = [col for col in dataset.columns if col not in targets]

# Handle categorical features
def cat_conv(data):
    data['Source IP'] = data['Source IP'].apply(hash).astype('float64')
    data[' Destination IP'] = data[' Destination IP'].apply(hash).astype('float64')
    return data

clean_data = cat_conv(dataset)

# Splitting data into X and y
def X_y_creation(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

X, y_multi = X_y_creation(clean_data)

# Handling missing and infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(axis=0, inplace=True)
y_multi = y_multi.loc[X.index].reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi_encoded, test_size=0.2, random_state=42, stratify=y_multi_encoded
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for Balancing Dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature selection using ExtraTreesClassifier with hyperparameter tuning
clf = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

# Ridge Regression-Based ELM
class ELM_Ridge:
    def _init_(self, hidden_units, activation_function, x, y, lambda_reg, elm_type, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.class_num = len(np.unique(self.y))
        self.lambda_reg = lambda_reg
        self.elm_type = elm_type

        if self.elm_type == 'clf':
            self.encoder = LabelEncoder()
            self.y_encoded = self.encoder.fit_transform(self.y)
            self.y_temp = np.eye(self.class_num)[self.y_encoded]  # One-hot encoding

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=-1, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=-1, high=1, size=(self.hidden_units, 1))
        else:  # normal distribution
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        H = np.dot(self.W, x.T) + self.b
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-H)), H
        elif self.activation_function == 'relu':
            return np.maximum(0, H), H
        elif self.activation_function == 'sin':
            return np.sin(H), H
        elif self.activation_function == 'tanh':
            return np.tanh(H), H
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, H) + 0.1 * np.minimum(0, H), H
        else:
            raise ValueError("Unsupported activation function")

    def __hidden2output(self, H):
        return np.dot(H.T, self.beta)

    def softmax(self, output):
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))  # Stability trick (subtract max)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize to get probabilities
        return probabilities

    def fit(self):
        start_time = time.time()
        H, _ = self.__input2hidden(self.x)

        # Ridge regression weight calculation
        H_t_H = np.dot(H, H.T) + self.lambda_reg * np.eye(H.shape[0])

        self.beta = np.dot(pinv(H_t_H), np.dot(H, self.y_temp))  # Using pseudo-inverse

        end_time = time.time()
        self.train_time = end_time - start_time

        train_output = self.__hidden2output(H)

        if self.elm_type == 'clf':
            predicted_labels = np.argmax(train_output, axis=1)
            self.train_score = np.mean(predicted_labels == self.y_encoded)

        return self.beta, self.train_score, self.train_time

    def predict(self, x):
        H, hidden_activations = self.__input2hidden(x)
        output = self.__hidden2output(H)

        # Calculate probabilities (assuming softmax activation for classification)
        probabilities = self.softmax(output)

        if self.elm_type == 'clf':
            return probabilities, H, hidden_activations, output
        return output

    def score(self, x, y):
        y_pred = self.predict(x)[0]  # Get probabilities for scoring
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
        return np.mean(y_pred_labels == y)

# Main Execution
if _name_ == "_main_":
    elm_ridge_model = ELM_Ridge(hidden_units=1200, activation_function='relu', x=X_train, y=y_train, lambda_reg=0.01, elm_type='clf')

    beta, train_score, train_time = elm_ridge_model.fit()
    print(f"Training Accuracy: {train_score * 100:.2f}%")
    print(f"Training Time: {train_time:.4f} seconds")

    test_score = elm_ridge_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_score * 100:.2f}%")

    # Predictions
    y_pred_probs, _, _, _ = elm_ridge_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Precision, Recall, F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
