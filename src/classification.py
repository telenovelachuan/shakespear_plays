import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import dump, load


processed_data = pd.read_csv("../data/processed/processed.csv")
processed_data.dropna(inplace=True)
le = LabelEncoder()
#processed_data = processed_data.apply(le.fit_transform)

'''
ColumnTransformation Pipeline
'''
# cat_attribs = ["Play", "PL_contain_!", "PL_contain_?", "ActSceneLine", "PlayerLine"]  # for non for one hot encoded
cat_attribs = ["Play", "PL_contain_!", "PL_contain_?", "ActSceneLine"]  # for one hot encoded
num_attribs = ["Play_length", "PlayerLinenumber", "PL_length", "ASL_1", "ASL_2", "ASL_3", "PL_w_count", "PL_w_density", "PL_#_comma_split", "PL_#_stop_words", "PL_#_upper_case"]
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    # ("cat", OneHotEncoder(), cat_attribs)
])
# col_to_remove = ["Player"] + cat_attribs
# processed_X = processed_data.drop(columns=col_to_remove, axis=1)  # for non one hot encoded
# processed_X = processed_data.drop(["Player", "PlayerLine"], axis=1)  # for one hot encoded
processed_data = processed_data.drop(["PlayerLine", "Dataline"], axis=1)
columns_to_dummy = processed_data.columns.to_list()
columns_to_dummy.remove("Player")
data_prepared = pd.get_dummies(processed_data, columns=columns_to_dummy)
# print "dummy:{}".format(pd.get_dummies(processed_data).info())
# data_prepared = full_pipeline.fit_transform(processed_data)
processed_y = processed_data["Player"]
print "data_prepared:{}".format(data_prepared.info)

# train/test split
train_set, test_set = train_test_split(data_prepared, test_size=0.2, random_state=42)

# col_to_remove = ["Player"]  # for non one hot encoded
col_to_remove = ["Player"]  # for one hot encoded
X_training = train_set.drop(columns=col_to_remove, axis=1)
X_test = test_set.drop(columns=col_to_remove, axis=1)
y_training = train_set["Player"]
y_test = test_set["Player"]


print "X_training:{}".format(X_training.info())
print "X_test: {}".format(X_test.info())
# print train_set.info()


def k_fold_cross_validation(model, training_data, training_labels, k):
    print "begin k-fold cross validation..."
    scores = cross_val_score(model, training_data, training_labels, scoring="accuracy", cv=k)
    print "scores: {}".format(scores)
    rmse_scores = np.sqrt(-scores)
    print "np.sqrt(-scores): {}".format(rmse_scores)
    print "Mean: {}".format(rmse_scores.mean())
    print "Standard deviation: {}".format(rmse_scores.std())
    print "Done."


def get_prediction_precision(model, testing_data, testing_labels):
    y_pred = model.predict(testing_data)
    print "y_pred:{}".format(y_pred)
    n_correct = sum(y_pred == testing_labels)
    print "n_correct:{}".format(n_correct)
    print "accuracy: {}".format(float(n_correct) / len(y_pred))

'''
Logistic Regression
'''
# print "begin training logistic regression..."
# print datetime.datetime.now()
# from sklearn.linear_model import LogisticRegression
# softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
# softmax_reg.fit(X_training, y_training)
# print "training finished, saving model..."
# print datetime.datetime.now()
# dump(softmax_reg, '../models/softmax.joblib')
# print "saving model finished, get predictions..."
# get_prediction_precision(softmax_reg, X_test, y_test)
# # k_fold_cross_validation(softmax_reg, X_training, y_training, 3)
# print "LR Done."
# #  accuracy: 0.5556

'''
Random Forest
'''
# from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(n_estimators=20)
# print "begin training random forest..."
# print datetime.datetime.now()
# rf_clf.fit(X_training, y_training)
# print "training finished, saving model..."
# print datetime.datetime.now()
# dump(rf_clf, '../models/rf.joblib')
# print "saving model finished, get predictions..."
# get_prediction_precision(rf_clf, X_test, y_test)
# print "score: {}".format(rf_clf.score(X_test, y_test))
# print "Done."
#
# k_fold_cross_validation(rf_clf, X_training, y_training, 3)
# # accuracy: 0.6132

'''
SVM
'''
print "begin training SVM..."
print datetime.datetime.now()
from sklearn.svm import SVC
svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)
svm_clf.fit(X_training, y_training)
print "training finished, saving model..."
print datetime.datetime.now()
dump(svm_clf, '../models/svm.joblib')
print "saving model finished, get predictions..."
get_prediction_precision(svm_clf, X_test, y_test)
# k_fold_cross_validation(softmax_reg, X_training, y_training, 3)
print "SVM Done."
#  accuracy: 0.5556

'''
ANN
'''