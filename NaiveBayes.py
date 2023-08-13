"""
@author: vovuncozer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Obtain x values and class labels for both datasets
train = pd.read_csv('bbcsports_train.csv')
x_train = train.iloc[:, :-1]
y_train = train[train.columns[-1]]

validation = pd.read_csv('bbcsports_val.csv')
x_val = validation.iloc[:, :-1]
y_val = validation[validation.columns[-1]]

#Obtain the number of occurrences of the word j in documents with label yk
T = train.groupby('class_label').sum()

#Obtain the total number of occurrences of all the words in documents with label yk
Tjyk = T.sum(axis='columns')
Tjyk = Tjyk.to_numpy()

T = T.to_numpy()

#Obtain total number of classes for both datasets
nyk = train.groupby('class_label')['class_label'].count()
nyk = nyk.to_numpy()
x_val = x_val.to_numpy()
nyk_val = validation.groupby('class_label')['class_label'].count()

#Obtain class probabailities
n_total = x_train.shape[0]

pi = []
for i in range (5):
    pi.append(nyk[i]/n_total)

classes = [0,1,2,3,4]

#Visualize the class distribution for both datasets
plt.bar(classes,nyk)
plt.title("Class Distribution in Training Dataset")
plt.xlabel("Classes")
plt.ylabel("Number of Classes")

plt.bar(classes,nyk_val)
plt.title("Class Distribution in Validation Dataset")
plt.xlabel("Classes")
plt.ylabel("Number of Classes")


#Build the model without a Dirichlet prior and evaluate on the validation dataset
prediction = []
for i in range (len(x_val)):
    arg = []    
    for c in classes:
        firstlog  = np.log(pi[c])
        sum1 = 0
        for j in range (np.shape(x_val)[1]):
            if T[c][j] == 0:
                secondlog =  x_val[i][j] * np.nan_to_num(-np.inf)
            else:
                secondlog = x_val[i][j] * (np.log((T[c][j])/(Tjyk[c])))  
            sum1 += secondlog
        arg.append(firstlog + sum1)  
    prediction.append(arg.index(max(arg)))

#Calculate the accuracy    
correct = 0
false = 0
for i in range (len(prediction)):
    if prediction[i] == y_val[i]:
        correct += 1
    else:
        false += 1

accuracy = correct/(correct+false)
print(accuracy)

#Calculate the confusion matrix   
confusion_matrix1 = np.zeros((5, 5))
for i in range(len(y_val)):
    confusion_matrix1[y_val[i]][prediction[i]] += 1

#Build the model with a Dirichlet prior and evaluate on the validation dataset
a = 1
prediction2 = []
for i in range (len(x_val)):
    arg = []    
    for c in classes:
        firstlog  = np.log(pi[c])
        sum1 = 0
        for j in range (np.shape(x_val)[1]):
            secondlog = x_val[i][j] * (np.log((T[c][j]+a)/(Tjyk[c]+(a*np.shape(x_val)[1]))))            
            sum1 += secondlog            
        arg.append(firstlog + sum1)  
    prediction2.append(arg.index(max(arg)))

correct2 = 0
false2 = 0
for i in range (len(prediction2)):
    if prediction2[i] == y_val[i]:
        correct2 += 1
    else:
        false2 += 1

#Calculate the accuracy with smoothing
accuracy2 = correct2/(correct2+false2)
print(accuracy2)

#Calculate the confusion matrix with smoothing
confusion_matrix2 = np.zeros((5, 5))
for i in range(len(y_val)):
    confusion_matrix2[y_val[i]][prediction2[i]] += 1

