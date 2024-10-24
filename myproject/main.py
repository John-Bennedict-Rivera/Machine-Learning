from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os


app = Flask(__name__)

# Global variables for the best model and vectorizer
best_model = None
vectorizer = None

# Path to the dataset in your directory
dataset_path = os.path.join(os.getcwd(), 'text_emotion_prediction.csv')

# Function to train models and get results
def train_models(df):
    global vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Comment'])  # Text data (features)
    y = df['Emotion']  # Emotion data (target labels)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate different models
    results = {}
    models = {}

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    results['Naive Bayes'] = round(accuracy_nb * 100, 1)
    models['Naive Bayes'] = nb

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    results['KNN'] = round(accuracy_knn * 100, 1)
    models['KNN'] = knn

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    results['Decision Tree'] = round(accuracy_dt * 100, 1)
    models['Decision Tree'] = dt

    # Support Vector Machine (SVM)
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    results['SVM'] = round(accuracy_svm * 100, 1)
    models['SVM'] = svm

    # Find the best model by accuracy
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]

    # Set the global best_model to the best-performing model
    global best_model
    best_model = models[best_model_name]

    return best_model_name, best_accuracy

# Load the dataset and train the models when the application starts
def load_and_train():
    global dataset_path
    df = pd.read_csv(dataset_path)

    # Print columns to verify if 'Comment' and 'Emotion' exist
    print("Columns in DataFrame:", df.columns)

    # Clean column names to avoid leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Check if 'Comment' exists in the DataFrame
    if 'Comment' not in df.columns or 'Emotion' not in df.columns:
        raise ValueError("The required columns are missing from the DataFrame.")

    # Train models and select the best one
    best_model_name, best_accuracy = train_models(df)
    print(f"Best Model: {best_model_name} with accuracy: {best_accuracy}%")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']  # Get the user's input sentence

        # Check if the vectorizer and best model are available
        if vectorizer is None or best_model is None:
            return "Vectorizer or model is not initialized. Please check the server setup."

        # Vectorize the sentence
        X_new = vectorizer.transform([sentence])

        # Use the best model to predict the emotion
        predicted_emotion = best_model.predict(X_new)

        # Display the result using result.html
        return render_template('result.html', sentence=sentence, emotion=predicted_emotion[0])
if __name__ == '__main__':
    # Automatically load the dataset and train the models when the app starts
    load_and_train()
    
    app.run(debug=True)