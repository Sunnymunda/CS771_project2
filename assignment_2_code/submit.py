import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Changed import
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words

# Dictionary file path
dictionary_file_path = '/content/dict'

# Load words from dictionary file
dictionary = load_dictionary(dictionary_file_path)

def generate_bigrams(word):
    return [word[i:i+2] for i in range(len(word)-1)]

def process_bigrams(bigrams):
    bigrams = sorted(set(bigrams))
    return bigrams[:5]

def generate_features(word):
    bigrams = generate_bigrams(word)
    processed_bigrams = process_bigrams(bigrams)
    return ' '.join(processed_bigrams)
features_list = [generate_features(word) for word in dictionary]
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\S+')
X = vectorizer.fit_transform(features_list).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dictionary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
knn_clf.fit(X, y)

def predict_top_words(input_word):
    # Generate features for the input word
    input_features = generate_features(input_word)
    input_vector = vectorizer.transform([input_features]).toarray()

    # Predict probabilities for the input vector
    probabilities = knn_clf.predict_proba(input_vector)[0]

    # Get the top 5 indices with the highest probabilities
    top_indices = np.argsort(probabilities)[-5:][::-1]

    # Get the corresponding top words
    top_words = label_encoder.inverse_transform(top_indices)
    return top_words

correct_count = 0
total_count = len(X_test)

for i in range(total_count):
    true_word = label_encoder.inverse_transform([y_test[i]])[0]
    predicted_top_words = predict_top_words(true_word)
    #print(predicted_top_words)
    #print(true_word)
    if true_word in predicted_top_words:
        correct_count += 1

accuracy = correct_count / total_count
print(f'Accuracy: {accuracy:.2f}')

predict_top_words('insufficient')

predict_top_words('productive')