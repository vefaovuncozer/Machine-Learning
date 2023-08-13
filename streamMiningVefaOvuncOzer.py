"""
@author: Vefa Övünç Özer
"""
#Import necessary libraries
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np 

#Generate data sets with given specifications and write them to a file with desired format
hp_a = HyperplaneGenerator(random_state=42, n_features=10, n_drift_features=2,noise_percentage=0.10)
hpa  = hp_a.next_sample(20000);
ta   = np.c_[ hpa[0], hpa[1]] 
np.savetxt(("Hyperplane Dataset 10_2.csv"), ta)

hp_b = HyperplaneGenerator(random_state=42,n_features=10, n_drift_features=2,noise_percentage=0.30)
hpb  = hp_b.next_sample(20000);
tb   = np.c_[ hpb[0], hpb[1]] 
np.savetxt(("Hyperplane Dataset 30_2.csv"), tb)

hp_c = HyperplaneGenerator(random_state=42,n_features=10, n_drift_features=5,noise_percentage=0.10)
hpc  = hp_c.next_sample(20000);
tc   = np.c_[ hpc[0], hpc[1]] 
np.savetxt(("Hyperplane Dataset 10_5.csv"), tc)

hp_d = HyperplaneGenerator(random_state=42,n_features=10, n_drift_features=5,noise_percentage=0.30)
hpd  = hp_d.next_sample(20000);
td   = np.c_[ hpd[0], hpd[1]] 
np.savetxt(("Hyperplane Dataset 30_5.csv"), td)

data_streams = [hp_a,hp_b,hp_c,hp_d]
data_stream_names = ["Hyperplane Dataset 10_2","Hyperplane Dataset 30_2","Hyperplane Dataset 10_5","Hyperplane Dataset 30_5"]
batch_sizes = [1,100,1000]


#Construct HoeffdingTree classifier and train for different datasets and batch sizes
HT  = HoeffdingTreeClassifier()
for i in range (4):
    for batch_size in batch_sizes:
        evaluate = EvaluatePrequential(max_samples = 20000, batch_size=batch_size, metrics=['accuracy'])
        evaluate.evaluate(stream = data_streams[i], model = [HT], model_names = ["HT"])
        print("Accuracy :" + str(data_stream_names[i]) + " Batch Size :" + str(batch_size))

#Construct KNN classifier and train for different datasets and batch sizes
KNN = KNNClassifier()
for i in range (4):
    for batch_size in batch_sizes:
        evaluate = EvaluatePrequential(max_samples = 20000, batch_size=batch_size, metrics=['accuracy'])
        evaluate.evaluate(stream = data_streams[i], model = [KNN], model_names = ["KNN"])
        print("Accuracy :" + str(data_stream_names[i]) + " Batch Size :" + str(batch_size))    

#Construct NB classifier and train for different datasets and batch sizes
NB  = NaiveBayes()
for i in range (4):
    for batch_size in batch_sizes:
        evaluate = EvaluatePrequential(max_samples = 20000, batch_size=batch_size, metrics=['accuracy'])
        evaluate.evaluate(stream = data_streams[i], model = [NB], model_names = ["NB"])
        print("Accuracy :" + str(data_stream_names[i]) + " Batch Size :" + str(batch_size))

#Plot the accuracy and mean of the online single classifiers using scikit-multiflow 
evaluate = EvaluatePrequential(max_samples = 20000, show_plot = True, metrics=['accuracy'])
evaluate.evaluate(stream = HyperplaneGenerator(), model = [HT, KNN, NB], model_names = ['HT', 'KNN','NB'])

"""
MajorityVoting functions predicts the class of the value 
using Major Voting rule.
Parameters: 
    HT, KNN, NB: The online classifiers
    stream : The data stream 
"""
def MajorityVoting(HT,KNN,NB,stream):
    corrects = 0
    for i in range (20000):
        a = np.array([],dtype=int)
        X, y = stream.next_sample()
        ht_pred  = HT.predict(X)
        knn_pred = KNN.predict(X)
        nb_pred  = NB.predict(X)
        a = np.append(a, ht_pred) 
        a = np.append(a, knn_pred) 
        a = np.append(a, nb_pred) 
        # All the predictions from different classifiers are added to array 
        # and the most frequent one is chosen as prediction
        y_pred = np.bincount(a).argmax()
        if y == y_pred:
            corrects += 1
        HT  = HT.partial_fit(X, y)
        KNN = KNN.partial_fit(X, y)
        NB  = NB.partial_fit(X, y)
    print('Overall accuracy: {}'.format(corrects / 20000))
    
#MajorityVoting is trained with different data streams 
MajorityVoting(HT,KNN,NB,hp_a)     
MajorityVoting(HT,KNN,NB,hp_b)
MajorityVoting(HT,KNN,NB,hp_c)
MajorityVoting(HT,KNN,NB,hp_d) 

"""
MajorityVoting functions predicts the class of the value 
using Major Voting rule,however, the weights are changing 
and peanlty applied accoring to the previous choice of the
classifier compared with the other 2 classifiers.
Parameters: 
    HT, KNN, NB: The online classifiers
    stream : The data stream 
"""
def WeightedMajorityVoting(HT,KNN,NB,stream):
    weights = [1/3,1/3,1/3]
    corrects = 0
    for i in range (20000):
        a = np.array([],dtype=int)
        X, y = stream.next_sample()
        ht_pred  = HT.predict(X)
        knn_pred = KNN.predict(X)
        nb_pred  = NB.predict(X)
        a = np.append(a, weights[0]*ht_pred) 
        a = np.append(a, weights[1]*knn_pred) 
        a = np.append(a, weights[2]*nb_pred)
        #Decides whether the class is 0 or 1 
        if np.sum(a) > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        # Penalty is applied such that wrong classifiers has less weight (decision power)
        # and other classifiers have awarded.
        if y_pred != ht_pred:
            if  weights[0] > 1/20000:
                weights[0] = weights[0]-(1/10000)
                weights[1] = weights[1]+(1/20000)
                weights[2] = weights[2]+(1/20000)
        elif y_pred != knn_pred:
            if weights [1] > 1/20000:
                weights[0] = weights[0]+(1/20000)
                weights[1] = weights[1]-(1/10000)
                weights[2] = weights[2]+(1/20000)
        elif y_pred != nb_pred:
            if  weights[2] > 1/20000:
                weights[0] = weights[0]+(1/20000)
                weights[1] = weights[1]+(1/20000)
                weights[2] = weights[2]-(1/10000)
        if y == y_pred:
            corrects += 1
        HT  = HT.partial_fit(X, y)
        KNN = KNN.partial_fit(X, y)
        NB  = NB.partial_fit(X, y)
    print('Overall accuracy: {}'.format(corrects / 20000))
    print(weights[0],weights[1],weights[2]) 
    
#WeightedMajorityVoting is trained with different data streams 
WeightedMajorityVoting(HT,KNN,NB,hp_a)
WeightedMajorityVoting(HT,KNN,NB,hp_b)
WeightedMajorityVoting(HT,KNN,NB,hp_c)
WeightedMajorityVoting(HT,KNN,NB,hp_d)


