#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df


# In[5]:


import numpy as np
from sklearn import svm

# Create a subset of the data with only 'ClassA' and 'ClassB' samples
selected_classes = [0, 1]
subset_data = df[df['Label'].isin(selected_classes)]

# Split the dataset into features (X) and labels (y)
X = subset_data[['embed_0', 'embed_1']]
y = subset_data['Label']

# Create an SVM classifier and train it on the selected data
clf = svm.SVC()
clf.fit(X, y)
# Get the support vectors
support_vectors = clf.support_vectors_

# Print the support vectors
print("Support Vectors:")
print(support_vectors)



# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# Your original dataset


# Split the data into features (X) and labels (y)
X = df[['embed_0', 'embed_']]
y = df['Label']

# Split the data into a training set and a test set (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier and train it on the training data
clf = svm.SVC()
clf.fit(X_train, y_train)

# Calculate and print the accuracy on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Perform classification for a given vector (test_vect)
test_vect = [[10, 5]]  # Example test vector
predicted_class = clf.predict(test_vect)
print("Predicted Class for test_vect:", predicted_class)


# In[30]:


# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['embed_0', 'embed_1']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['embed_0', 'embed_1']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[31]:


# Assuming you have already trained the SVM classifier and have the test set (X_test and y_test)

# Predict the class labels for the test set
predicted_labels = clf.predict(X_test)

# Compare the predicted labels with the true labels and calculate accuracy manually
correct_predictions = 0
total_samples = len(y_test)

for true_label, predicted_label in zip(y_test, predicted_labels):
    if true_label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / total_samples

print("Manually Calculated Accuracy:", accuracy)


# In[36]:


import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

# Your original dataset

# Split the data into features (X) and labels (y)
X = df[['embed_0', 'embed_1']]
y = df['Label']

# Split the data into a training set and a test set (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier and train it on the training data
clf = svm.SVC()
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test set
predicted_classes = clf.predict(X_test)

# Study the output values and relate them to class values
for i, predicted_class in enumerate(predicted_classes):
    print(f"Predicted: {predicted_class}, Actual: {y_test.iloc[i]}")

# Test the accuracy of the SVM classifier
correct_predictions = sum(predicted_classes == y_test)
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'data' is your DataFrame or dictionary
X = df[['embed_0', 'embed_1']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

# Iterate through different kernel functions
for kernel in kernel_functions:
    # Initialize the SVC classifier with the specified kernel
    clf = SVC(kernel=kernel)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Use the trained classifier to make predictions on the test set
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Display the results for each kernel function
    print(f"Kernel: {kernel}, Accuracy: {accuracy}")




