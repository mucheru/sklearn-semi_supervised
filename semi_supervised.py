#define dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading

X,y = make_classification(n_samples=1000,n_features = 2,n_informative=2,n_redundant=0,random_state=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.5,random_state=1,stratify=y)
X_train_lab,X_test_unlab,y_train_lab,y_test_unlab = train_test_split(X_train,y_train,test_size=0.50,random_state=1,stratify=y_train)
print('Labeled Train set:',X_train_lab.shape,y_train_lab.shape)
print('Unlabled train set:',X_test_unlab.shape,y_test_unlab.shape)
#summarize test set size
print('Test set:',X_test.shape,y_test.shape)
#model = LogisticRegression()
model = LabelSpreading()
#fit the on labeled dataset
model.fit(X_train_lab,y_train_lab)
#make prediction on test set
yhat=model.predict(X_test)
#calculate score for the test set
score=accuracy_score(y_test,yhat)
# summarize score
print('Accuracy :%.3f' % (score*100))
