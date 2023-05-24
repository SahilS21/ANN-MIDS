import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('IMDB_Dataset.csv')
data

# Split the data into training and testing sets
X = data['review']  # Document text
y = data['sentiment']  # Positive or negative label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer= CountVectorizer()
x_train_vectorized= vectorizer.fit_transform(X_train)

model= MultinomialNB()
model.fit(x_train_vectorized,y_train)

x_test_vectorized= vectorizer.transform(X_test)
accuracy= model.score(x_test_vectorized, y_test)
print(accuracy*100)

new_doc= ["The college is very good"]
new_doc_vec= vectorizer.transform(new_doc)
print(model.predict(new_doc_vec))