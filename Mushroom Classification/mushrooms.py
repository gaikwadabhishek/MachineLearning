import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

#Read the data using pandas
data=pd.read_csv("mushrooms.csv")
print(data.head())
print(data.describe().transpose())
total_null_values = sum(data.isnull().sum())
print(total_null_values)
from sklearn.preprocessing import LabelEncoder
Enc = LabelEncoder()
data_tf = data.copy()
for i in data.columns:
    data_tf[i]=Enc.fit_transform(data[i])

print(data_tf)
X = data_tf.drop(['class'], axis=1)
Y = data_tf['class']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

X_train.head(5)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_clf = LogisticRegression()
log_clf.fit(X_train, Y_train)
X_pred = log_clf.predict(X_train)
Y_pred = log_clf.predict(X_test)

print(accuracy_score(Y_test, Y_pred))

from sklearn import metrics
print("Training accuracy:",metrics.accuracy_score(Y_train, X_pred))

#use the trained model to predict the test values
test_predict = log_clf.predict(X_test)
print("Testing accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("\nClassification Report for LogisticRegression:")
print(metrics.classification_report(Y_test, Y_pred, target_names=['edible','poisonous']))
print("\nConfusion Matrix:")
skcm = metrics.confusion_matrix(Y_test,Y_pred)
#putting it into a dataframe so it prints the labels
skcm = pd.DataFrame(skcm, columns=['predicted-edible','predicted-poisonous'])
skcm['actual'] = ['edible','poisonous']
skcm = skcm.set_index('actual')

#NOTE: NEED TO MAKE SURE I'M INTERPRETING THE ROWS & COLS RIGHT TO ASSIGN THESE LABELS!
print(skcm)

print("\nScore: ", log_clf.score(X_test,Y_test))


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, Y_train)
X_pred = rnd_clf.predict(X_train)
Y_pred = rnd_clf.predict(X_test)
accuracy_score(Y_test, Y_pred)


from sklearn import metrics
print("Training accuracy:",metrics.accuracy_score(Y_train, X_pred))

#use the trained model to predict the test values
test_predict = rnd_clf.predict(X_test)
print("Testing accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("\nClassification Report for RandomForestClassifier:")
print(metrics.classification_report(Y_test, Y_pred, target_names=['edible','poisonous']))
print("\nConfusion Matrix:")
skcm = metrics.confusion_matrix(Y_test,Y_pred)
#putting it into a dataframe so it prints the labels
skcm = pd.DataFrame(skcm, columns=['predicted-edible','predicted-poisonous'])
skcm['actual'] = ['edible','poisonous']
skcm = skcm.set_index('actual')

#NOTE: NEED TO MAKE SURE I'M INTERPRETING THE ROWS & COLS RIGHT TO ASSIGN THESE LABELS!
print(skcm)

print("\nScore: ", rnd_clf.score(X_test,Y_test))

from sklearn import preprocessing
oh = preprocessing.OneHotEncoder(categorical_features='all')
oh.fit(X)
xo = oh.transform(X).toarray()

print('\nExample Feature Values - row 1 in X:')
print(X.iloc[1])
print('\nExample Encoded Feature Values - row 1 in xo:')
print(xo[1])
print('\nClass Values (Y):')
print(np.array(Y))
#print('\nEncoded Class Values (y):')
#print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
#initialize and fit the naive bayes classifier
from sklearn.naive_bayes import GaussianNB
skgnb = GaussianNB()
skgnb.fit(x_train,y_train)
train_predict = skgnb.predict(x_train)
#print(train_predict)

#see how accurate the training data was fit
from sklearn import metrics
print("Training accuracy:",metrics.accuracy_score(y_train, train_predict))

#use the trained model to predict the test values
test_predict = skgnb.predict(x_test)
print("Testing accuracy:",metrics.accuracy_score(y_test, test_predict))


print("\nClassification Report for Gaussian Naive Bayes:")
print(metrics.classification_report(y_test, test_predict, target_names=['edible','poisonous']))
print("\nConfusion Matrix:")
skcm = metrics.confusion_matrix(y_test,test_predict)
#putting it into a dataframe so it prints the labels
skcm = pd.DataFrame(skcm, columns=['predicted-edible','predicted-poisonous'])
skcm['actual'] = ['edible','poisonous']
skcm = skcm.set_index('actual')

#NOTE: NEED TO MAKE SURE I'M INTERPRETING THE ROWS & COLS RIGHT TO ASSIGN THESE LABELS!
print(skcm)

print("\nScore: ", skgnb.score(x_test,y_test))


