A data science project on the Shakespear plays dataset to explore more useful features and build classification models to determin the player using other columns as features.


# Feature explorative ideas
Try to extract info using NLTK from "PlayerLine" since it's text, and apply similar metrics on categorical columns like "Play" and "ActSceneLine"
- split ActSceneLine by dot
- add total number of words for PlayerLine
- add total number of characters for PlayerLine
- calculate the length of Play
- whether PlayerLine contains exclamation mark
- whether PlayerLine contains question mark
- number of parts in PlayerLine splitted by comma
- number of stop words in PlayerLine 
- number of upper case letters in PlayerLine
- PlayerLine word density - average length of words used in PlayerLine


# Data preprocessing
- load processed dataset, with all 10 explorative features added, into Pandas
- for numeric features:
  1. use imputer that fills median values
  2. use StandardScaler to normalize feature values
- for categorical features: use one hot encoding to encode feature values
- split train/test data via random shuffling
- drop Player column to form training data
- for Keras Dense Neural Network, one-hot encode the label in advance and transform to matrix


# Classification process to determine Player using other features
models tried:

1. Logistic Regression

Uses multinomial multi_class regressor in sklearn(softmax) to conduct multi-label classification. It trains rather slow due to the huge amount of attributes.
Reaches an accuracy of 0.5556 on testing set.

2. Random Forest

Tuned n_estimators to be 20. Trained much faster than LR, and reaches an accuracy of 0.6132 on testing set.
Further increasing the number of trees does not help increase accuracy.

3. SVM

Uses the Support Vector Classifier with a "poly" kernel by sklearn. Training takes too much time(more than several days) and never converges...
I think the huge amount of attributes(especially the sparse matrices by one-hot encoding) slows down SVM's kernel trick to find a linear-separable solution in high dimensions.
Besides, the SVC class implemented by sklearn is not specifically optimized such as LinearSVC does.

4. Neural Network

Uses Keras to construct and compile a neural network that contains 2 dense hidden layers and 1 output layer(more layers barely helps).
"RELU" is used for activation inside hidden layers, and "softmax" for the final output layer.
60 epochs' training achieves an accuracy of 0.741 on testing set. Training for more epochs could further improve accuracy, but increases overfitting as well.
