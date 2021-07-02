'''
The following code is for the K-Means
Created by - ANALYTICS VIDHYA
'''

# importing required libraries
import pandas as pd
from sklearn.cluster import KMeans

# read the train and test dataset
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')

# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)

# Now, we need to divide the training data into differernt clusters
# and predict in which cluster a particular data point belongs.  

'''
Create the object of the K-Means model
You can also add other parameters and test your code here
Some parameters are : n_clusters and max_iter
Documentation of sklearn KMeans: 

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
 '''

model = KMeans()  

# fit the model with the training data
model.fit(train_data)

# Number of Clusters
print('\nDefault number of Clusters : ',model.n_clusters)

# predict the clusters on the train dataset
predict_train = model.predict(train_data)
print('\nCLusters on train data',predict_train) 

# predict the target on the test dataset
predict_test = model.predict(test_data)
print('Clusters on test data',predict_test) 

# Now, we will train a model with n_cluster = 3
model_n3 = KMeans(n_clusters=3)

# fit the model with the training data
model_n3.fit(train_data)

# Number of Clusters
print('\nNumber of Clusters : ',model_n3.n_clusters)

# predict the clusters on the train dataset
predict_train_3 = model_n3.predict(train_data)
print('\nCLusters on train data',predict_train_3) 

# predict the target on the test dataset
predict_test_3 = model_n3.predict(test_data)
print('Clusters on test data',predict_test_3) 