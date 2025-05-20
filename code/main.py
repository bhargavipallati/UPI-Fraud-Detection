import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
print("==================================================")
print("Malware Dataset")
print(" Process - Malware Attack Detection")
print("==================================================")
##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("training set.csv")
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()
------------------------------------------PRE PROCESSING--------------------------------------------------
#checking missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print()
print("----------------------------------------------")
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
------------------------------------------DATA SPLITTING---------------------------------------------------
df_train_y=dataframe_2["label"]
df_train_X=dataframe_2.iloc[:,:20]
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_train_X['proto'] = number.fit_transform(df_train_X['proto'].astype(str))
df_train_X['service'] = number.fit_transform(df_train_X['service'].astype(str))
df_train_X['state'] = number.fit_transform(df_train_X['state'].astype(str))
#df_train_X['attack_cat'] = number.fit_transform(df_train_X['attack_cat'].astype(str))
print("--------------------------------------------------")
print(" Preprocessing")
print("==================================================")

df_train_X.head(5)
x=df_train_X
y=df_train_y

--------------------------------------FEATURE SELECTION------------------------------------------------
---------------------------------------------KMEANS------------------------------------------------------------
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y_true = make_blobs(n_samples=175341, centers=4,cluster_std=0.30, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=20);

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=20, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title("k-means")

plt.show()
#---------------------------------------------------------------------------------------
import scipy.cluster.hierarchy
from pyxdameraulevenshtein import damerau_levenshtein_distance
def cluster_ngrams(ngrams, compute_distance, max_dist, method):
   indices = np.triu_indices(len(ngrams), 1)
   pairwise_dists = np.apply_along_axis( lambda col: compute_distance(ngrams[col[0]], ngrams[col[1]]), 0, indices)
   hierarchy = scipy.cluster.hierarchy.linkage(pairwise_dists, method=method)
   clusters = dict((i, [i]) for i in range(len(ngrams)))
   for (i, iteration) in enumerate(hierarchy):
      cl1, cl2, dist, num_items = iteration
      if dist > max_dist:
        break
       items1 = clusters[cl1]
      items2 = clusters[cl2]
  del clusters[cl1]
 del clusters[cl2]
 clusters[len(ngrams) + i] = items1 + items2
 ngram_clusters = []
 for cluster in clusters.values():
    ngram_clusters.append([ngrams[i] for i in cluster])
    return ngram_clusters
   def dl_ngram_dist(ngram1, ngram2):
      return sum(damerau_levenshtein_distance(w1, w2) for w1, w2 in zip(ngram1, ngram2))
     x_train,x_test,y_train,y_test = train_test_split(df_train_X,y_kmeans,test_size = 0.20,random_state = 42)
x_train,x_test,y_train,y_test = train_test_split(df_train_X,y,test_size = 0.20,random_state = 42)
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100) 
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100

from sklearn.metrics import confusion_matrix
print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns 
sns.heatmap(cm2, annot = True, cmap ='plasma', linecolor ='black', linewidths = 1)
plt.show() 
#------------------------------
---------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)
print()
print("Decision Tree")
print()
Result_2=accuracy_score(y_test, dt_prediction)*100
print(metrics.classification_report(y_test,dt_prediction))
print()
print("DT Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma', linecolor ='black', linewidths = 1)
plt.show()
#--------------------------------- ROC GRAPH ---------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.get_params()
gradient_booster.fit(x_train,y_train)
gb_prediction = gradient_booster.predict(x_test)

print(classification_report(y_test,gradient_booster.predict(x_test)))
Result_2=accuracy_score(y_test, gb_prediction)*100
print()
print("gradient_booster Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
#------------------------------------ NAVIE BAYIES ------------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

----------------------------MAKING THE CONFUSION MATRIX---------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Navie Bayies Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred)
print(cm1)
print("-------------------------------------------------------")
print()
#------------------------------SVM ALGORITHM------------------------------------------------------------
"SVM Algorithm "
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.svm import SVC
svclassifier = SVC()
svclassifier.fit(x_train,y_train)
y_pred11 = svclassifier.predict(x_test)

result = confusion_matrix(y_test, y_pred11)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred11)
print("Classification Report:",)

print (result1)
print("Accuracy:",accuracy_score(y_test, y_pred11))
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(result, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Attack', 'Benign']); ax.yaxis.set_ticklabels(['Attack', 'Benign']);

#---------------------------------------------------------------------------------
inp=int(input('Enter the Malware Type'))
if (rf_prediction[inp] ==0 ):
   print("Ransomware ")
  elif (rf_prediction[inp] ==1 ):
     print("Ransomware ")
else:
   print("Spyware")






   


