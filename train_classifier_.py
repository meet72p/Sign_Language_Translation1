import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open('./dataset.p', 'rb'))

# Ensure all feature vectors have exactly 42 features
filtered_data = [item for item in data_dict['data'] if len(item) == 42]
filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 42]

# Print information
print(f"Total samples before filtering: {len(data_dict['data'])}")
print(f"Total samples after filtering: {len(filtered_data)}")

# Convert to NumPy array
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully!")
