import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import dump, load


processed_data = pd.read_csv("../data/processed/processed.csv")
processed_data.dropna(inplace=True)
X = processed_data.loc[:, processed_data.columns != 'Player']
y = processed_data['Player']
train_set, test_set = train_test_split(processed_data, test_size=0.2, random_state=42)
cat_attribs = ["Play", "PL_contain_!", "PL_contain_?", "ActSceneLine", "PlayerLine"]
num_attribs = ["Play_length", "PlayerLinenumber", "PL_length", "ASL_1", "ASL_2", "ASL_3", "PL_w_count", "PL_w_density", "PL_#_comma_split", "PL_#_stop_words", "PL_#_upper_case"]
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    # ("cat", OneHotEncoder(), cat_attribs)
])
col_to_remove = ["Player"] + cat_attribs
processed_X = train_set.drop(columns=col_to_remove, axis=1)
# processed_X = processed_X.drop("PlayerLine", axis=1)
data_prepared = full_pipeline.fit_transform(processed_X)
processed_y = train_set["Player"]

processed_X_test = test_set.drop(columns=col_to_remove, axis=1)
# processed_X_test = processed_X_test.drop("PlayerLine", axis=1)
data_prepared_test = full_pipeline.fit_transform(processed_X_test)
processed_y_test = test_set["Player"]


print "data_prepared:{}".format(data_prepared[0])
print "data_prepared_test: {}".format(data_prepared_test[0])
# print train_set.info()


def k_fold_cross_validation(model, training_data, training_labels, k):
    scores = cross_val_score(model, training_data, training_labels, scoring="neg_mean_squared_error", cv=k)
    rmse_scores = np.sqrt(-scores)
    print "Scores: {}".format(rmse_scores)
    print "Mean: {}".format(rmse_scores.mean())
    print "Standard deviation: {}".format(rmse_scores.std())


def get_prediction_precision(model, testing_data, testing_labels):
    y_pred = model.predict(testing_data)
    print "y_pred:{}".format(y_pred)
    n_correct = sum(y_pred == testing_labels)
    print "n_correct:{}".format(n_correct)
    print "accuracy: {}".format(float(n_correct) / len(y_pred))

'''
Logistic Regression
'''
# print "begin model training..."
# import datetime
# print datetime.datetime.now()
# from sklearn.linear_model import LogisticRegression
# softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
# softmax_reg.fit(data_prepared, processed_y)
# print "training finished, saving model..."
# print datetime.datetime.now()
# dump(softmax_reg, 'softmax.joblib')
# print "saving model finished, get predictions..."
# get_prediction_precision(softmax_reg, processed_X_test, processed_y_test)
#k_fold_cross_validation(lin_reg, processed_X, processed_y, 10)
print "predict base on trained model."
softmax_reg = load('softmax.joblib')
get_prediction_precision(softmax_reg, data_prepared_test, processed_y_test)
# y = model.predict([data_prepared_test[0]])
# print y
print "Done."
'''
Random Forest
'''


'''
SVM
'''


'''
ANN
'''