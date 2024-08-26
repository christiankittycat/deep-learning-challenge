# deep-learning-challenge
Alphabet Soup Charity Model

This project aims to build a binary classification model to predict the success of organizations funded by Alphabet Soup using a neural network.

Steps Covered

1. Data Preprocessing

	•	Loaded the Dataset: The data was read from a CSV file containing information on various funded organizations.
	•	Dropped Irrelevant Columns: Removed non-beneficial columns (EIN and NAME) to simplify the dataset.
	•	Categorical Encoding: Encoded categorical variables using pd.get_dummies().
	•	Handling Rare Categories:
	•	For APPLICATION_TYPE and CLASSIFICATION columns, rare categories were combined into a new value “Other” to reduce noise.
	•	Feature Scaling: Used StandardScaler to scale the features for better model performance.

2. Model Building and Training

	•	Model Definition: Created a neural network with TensorFlow and Keras, consisting of two hidden layers and an output layer for binary classification.
	•	Model Compilation: Compiled the model using the Adam optimizer and binary cross-entropy loss function.
	•	Model Training: Trained the model for 100 epochs, tracking loss and accuracy to monitor the model’s learning process.

3. Model Evaluation

	•	Evaluated the Model: Used test data to evaluate the model’s performance and calculate the final loss and accuracy.

4. Model Saving

	•	Saved the Model: The trained model was saved as AlphabetSoupCharity.h5 for future use.

Usage

To run the model:

	1.	Preprocess the data as described.
	2.	Define and compile the neural network model.
	3.	Train the model and evaluate its performance.
	4.	Save the model for future predictions.

Dependencies

	•	Python 3.x
	•	TensorFlow and Keras
	•	Pandas
	•	Scikit-learn
