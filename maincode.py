
"Import Libaries "
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics


print("==================================================")
print("High Utility Pattern Mining Algorithm  Dataset")
print(" Process - High Utility Pattern Mining Algorithm ")
print("==================================================")


##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("/Users/akhil/OneDrive/Desktop/upi fraud detection using machine learning/Sourcecode/dataset.csv")
dataframe=dataframe.loc[:10000]
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()


 #2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
 

#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()

#3.Data splitting--------------------------------------------------- 

df_train_X=dataframe_2
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_train_X['CustomerID'] = number.fit_transform(df_train_X['CustomerID'].astype(str))
df_train_X['TransactionID'] = number.fit_transform(df_train_X['TransactionID'].astype(str))
df_train_X['CustomerDOB'] = number.fit_transform(df_train_X['CustomerDOB'].astype(str))
df_train_X['CustLocation'] = number.fit_transform(df_train_X['CustLocation'].astype(str))
df_train_X['TransactionDate'] = number.fit_transform(df_train_X['TransactionDate'].astype(str))
df_train_X['CustGender'] = number.fit_transform(df_train_X['CustGender'].astype(str))

print("==================================================")
print(" Preprocessing")
print("==================================================")

df_train_X.head(5)
x=df_train_X
y=df_train_X['CustGender']    


#---------------------------------------------------------------------------------------
x_train,x_test,y_train,y_test = train_test_split(df_train_X,y,test_size = 0.20,random_state = 42)
from sklearn import svm
classifier = svm.SVC()


classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
Result_1=accuracy_score(y_test, y_pred)*100

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("SVM Accuracy is:",Result_1,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred)
print(cm1)
print("-------------------------------------------------------")
print()
#------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100)  
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_2=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()



#---------------------------------------------------------------------------------------------

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=10,
                         learning_rate=20)

abc.fit(x_train, y_train)
AD_prediction=abc.predict(x_test)
print()
print("---------------------------------------------------------------------")
print("Ada Boost Algorithm ")
print()
Result_3=accuracy_score(y_test, AD_prediction)*100
print(metrics.classification_report(y_test,AD_prediction))
print()
print("Ada Boost  Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, AD_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph

#-----------------------------------------------------------------------------------
"Accuracy plot" 
list1=[Result_1,Result_2,Result_3]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['SVM ', 'random Forest ', 'Ada boost ']
students = list1
ax.bar(langs,students)
plt.show()

