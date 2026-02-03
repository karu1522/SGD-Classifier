# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset.Convert it into a pandas DataFrame and separate features X and labels y.
2. Divide the data into training and test sets using train_test_split.
3. Initialize the SGDClassifier with a maximum number of iterations.Fit the model on the training data.
4. Predict labels for the test set.Calculate accuracy score and display the confusion matrix.
Program:

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Karthic U
RegisterNumber: 212224040151 
```

## Head and Tail values:
```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
display(df.head())
display(df.tail())
X=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
```
## Output:
<img width="939" height="536" alt="image" src="https://github.com/user-attachments/assets/ddb45e2d-9d65-4431-8ae2-32043d4293b9" />


## Predicted values:
```py
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

sgd_clf.fit(x_train,y_train)

y_pred=sgd_clf.predict(x_test)

y_pred
```

## Output:
<img width="775" height="55" alt="image" src="https://github.com/user-attachments/assets/ee3e06ae-ebf9-44b5-b9da-42e6cd0074b4" />


## Accuracy:
```py
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy : {accuracy:.3f}")
```

## Output:
<img width="217" height="32" alt="image" src="https://github.com/user-attachments/assets/439bd468-9135-4dc8-a3be-009fc69e64c7" />


## Confusion matrix:
```py
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(cm)
```

## Output:
<img width="242" height="100" alt="image" src="https://github.com/user-attachments/assets/d847d33f-1f90-4683-b902-d5766292c195" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
