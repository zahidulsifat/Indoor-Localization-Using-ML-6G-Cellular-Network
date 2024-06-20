import pandas as pd

# Load your data into the 'results' DataFrame
results = pd.read_csv('Filename.csv')

# Define a threshold for acceptable distance error
threshold = 0.1

# Calculate the number of predictions with distance error less than or equal to the threshold
accurate_predictions = results[results['Distance_Error'] <= threshold]

# Calculate accuracy as a percentage
accuracy_percentage = (len(accurate_predictions) / len(results)) * 100

print("Accuracy: {:.2f}%".format(accuracy_percentage))
