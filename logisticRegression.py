

"""
LOGISTIC REGRESSION
"""

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = np.genfromtxt('dataset.csv', delimiter=',')
data = np.delete(data, (0), axis=0)
x = data[:,:-1]
y = data[:, [-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 3/10, random_state = 42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 2/3, random_state = 42)

def accuracy(x_test,y_test,weights,bias):
    prediction = []
    for j in range(len(y_test)):      
        sigma = np.exp(bias + np.dot(x_test[j],weights))
        p0 = float(1/(1+sigma))
        p1 = float(sigma/(1+sigma))
        if p1 > p0:
            prediction.append(1)
        else:
            prediction.append(0)   
    
    acc = accuracy_score(y_test, prediction)*100
    return prediction,acc

def StochasticGradientAscent(x,y,weights,bias,alpha,epoch):
    accurracy1 =[]    
    for k in range(epoch):
        for j in range(len(x)):
            sigma = float(np.exp(bias + np.dot(x[j],weights)))
            prediction = sigma/(1+sigma)
            weights = weights + alpha * x[j].reshape(12,1) * (float(y[j])-prediction)
            bias = bias + alpha*(float(y[j])-prediction)
        p,acc = accuracy(x,y,weights,bias)
        accurracy1.append(acc)    
    return weights,bias,accurracy1


def MiniBatchGradientAscent(x,y,weights,bias,alpha,epoch,batch_size):
    accurracy2 =[]   
    for k in range(epoch):
        for j in range((len(x)-64)//batch_size):
            batch_total = 0
            batch_totalbias = 0
            for l in range (j*64,(j+1)*64):
                sigma = np.exp(bias + np.dot(x[l],weights))
                prediction = sigma[0]/(1+sigma[0])
                
                error = x[l] * (y[l]-prediction)
                batch_total = batch_total + error
                
                errorbias = (y[l]-prediction)
                batch_totalbias = batch_totalbias + errorbias
           
            desc = batch_total
            descbias = batch_totalbias
            
            weights = weights + alpha * desc.reshape(12,1)
            bias = bias + alpha * descbias[0]
        p,acc = accuracy(x,y,weights,bias)
        accurracy2.append(acc) 
    return weights,bias, accurracy2


def FullBatchGradientAscent(x,y,weights,bias,alpha,epoch):
    accurracy3 =[] 
    for k in range(epoch):
            batch_total = 0
            batch_totalbias = 0
            for l in range (len(x)):
                if np.exp(bias + np.dot(x[l],weights))!=np.inf:
                    sigma = float(np.exp(bias + np.dot(x[l],weights)))
                    prediction = sigma/(1+sigma)
                    
                    error = x[l].reshape(12,1) * (float(y[l])-prediction)
                    batch_total = batch_total + error
                    
                    errorbias = (float(y[l])-prediction)
                    batch_totalbias = batch_totalbias + errorbias
           
            desc = batch_total
            descbias = batch_totalbias
            
            weights = weights + alpha * desc.reshape(12,1)
            bias = bias + alpha * descbias
            p,acc = accuracy(x,y,weights,bias)
            accurracy3.append(acc) 
    return weights,bias, accurracy3

# mu, sigma = 0, 1
# weights = np.random.normal(mu, sigma, 12)
# weights = weights.reshape(12,1)
# bias = 1
# weights,bias,accurracy1 = MiniBatchGradientAscent(x_train,y_train,weights,bias,10**-3,100,64)
# predictio1,acc = accuracy(x_test,y_test,weights,bias)
# print("MiniBatchGradientAscent1",acc)  

# weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# weights = weights.reshape(12,1)
# bias = 1
# weights,bias,accurracy2 = MiniBatchGradientAscent(x_train,y_train,weights,bias,10**-3,100,64)
# predictio1,acc = accuracy(x_test,y_test,weights,bias)
# print("MiniBatchGradientAscent2",acc)  

# weights = np.random.uniform(0.05, 0.05,12)
# weights = weights.reshape(12,1)
# bias = 1
# weights,bias,accurracy3 = MiniBatchGradientAscent(x_train,y_train,weights,bias,10**-3,100,64)
# predictio1,acc = accuracy(x_test,y_test,weights,bias)
# print("MiniBatchGradientAscent3",acc)  



# weights,bias,accurracy1 = StochasticGradientAscent(x_train,y_train,weights,bias,10**-3,100)
# predictio,acc, = accuracy(x_test,y_test,weights,bias)
# print("StochasticGradientAscent",acc)      
mu, sigma = 0, 1
weights = np.random.normal(mu, sigma, 12)
weights = weights.reshape(12,1)
bias = 1
weights,bias,accurracy2 = MiniBatchGradientAscent(x_train,y_train,weights,bias,10**-3,100,64)
predictio1,acc = accuracy(x_val,y_val,weights,bias)
print("MiniBatchGradientAscent",acc)   
#Calculate the confusion matrix   

confusion_matrix1 = np.zeros((2,2))
for i in range(len(y_val)):
    y = int(y_test[i])
    confusion_matrix1[y][predictio1[i]] += 1

# weights,bias,accurracy3 = FullBatchGradientAscent(x_train,y_train,weights,bias,10**-3,100)
# predictio,acc = accuracy(x_test,y_test,weights,bias)
# print("FullBatchGradientAscent",acc)
# epoch_plot =[]
# for i in range(100):
#     epoch_plot.append(i)
# colors=["r","g","b","c"]  
# learning_rates = [1,10**-3,10**-4,10**-5]
# for alpha, color in zip(learning_rates, colors):
#     weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     weights = weights.reshape(12,1)
#     bias = 1
#     weights,bias,accurracy1 = MiniBatchGradientAscent(x_train,y_train,weights,bias,alpha,100,64)
#     predictio1,acc = accuracy(x_test,y_test,weights,bias)
#     print("MiniBatchGradientAscent",acc)
#     plt.plot(epoch_plot, accurracy1, color=color, label='MiniBatchGradientAscent'+str(learning_rates))

# plt.plot(epoch_plot, accurracy1, color='r', label='MiniBatchGradientAscent-Gaussian')
# plt.plot(epoch_plot, accurracy2, color='g', label='MiniBatchGradientAscent-Zeros')
# plt.plot(epoch_plot, accurracy3, color='b', label='FullBatchGradientAscent-Uniform')
# plt.plot(epoch_plot, accurracy4, color='r', label='MiniBatchGradientAscent-Gaussian')
# plt.plot(epoch_plot, accurracy5, color='g', label='MiniBatchGradientAscent-Zeros')

plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Value")
plt.legend()
plt.show()
