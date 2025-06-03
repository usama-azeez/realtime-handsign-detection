import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Predict on test set
y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f'Model Accuracy: {accuracy}')

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print('Model saved successfully.')
