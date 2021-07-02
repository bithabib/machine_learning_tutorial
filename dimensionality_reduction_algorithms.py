'''
The following code is for Principal Component Analysis (PCA)
Created by - ANALYTICS VIDHYA
'''
# importing required libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  

# read the train and test dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# view the top 3 rows of the dataset
print(train_data.head(3))

# shape of the dataset
print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

# seperate the independent and target variable on training data
# target variable - Item_Outlet_Sales
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']

print('\nTraining model with {} dimensions.'.format(train_x.shape[1]))

# create object of model
model = LinearRegression()

# fit the model with the training data
model.fit(train_x,train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)

# Accuray Score on train dataset
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)

# Accuracy Score on test dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)

# create the object of the PCA (Principal Component Analysis) model
# reduce the dimensions of the data to 12
'''
You can also add other parameters and test your code here
Some parameters are : svd_solver, iterated_power
Documentation of sklearn PCA:

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
'''
model_pca = PCA(n_components=12)

new_train = model_pca.fit_transform(train_x)
new_test  = model_pca.fit_transform(test_x)

print('\nTraining model with {} dimensions.'.format(new_train.shape[1]))

# create object of model
model_new = LinearRegression()

# fit the model with the training data
model_new.fit(new_train,train_y)

# predict the target on the new train dataset
predict_train_pca = model_new.predict(new_train)

# Accuray Score on train dataset
rmse_train_pca = mean_squared_error(train_y,predict_train_pca)**(0.5)
print('\nRMSE on new train dataset : ', rmse_train_pca)

# predict the target on the new test dataset
predict_test_pca = model_new.predict(new_test)

# Accuracy Score on test dataset
rmse_test_pca = mean_squared_error(test_y,predict_test_pca)**(0.5)
print('\nRMSE on new test dataset : ', rmse_test_pca)