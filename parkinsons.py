#ProjectGurukul

#load all modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#read csv file of parkison's dataset
df=pd.read_csv("parkinsons_100k.csv")
#extract features but we don't want 'name' and 'status' columns
#so we will drop these two columns and store remainings
#axis 1 is for columns 
features = df.drop(['name','status'],axis=1)
#our target will be 'status'
target = df.loc[:,'status']

#scale all the datas in the range between -1,1
scaler= MinMaxScaler((-1,1))
features_c=scaler.fit_transform(features)

#split the dataset into training and testing sets where 20% data for testing purpose.
x_train,x_test,y_train,y_test=train_test_split(features_c, target , test_size=0.2,random_state=10)

#initialize the random forest classifier and fit the datas
model= RandomForestClassifier(random_state=2)
model.fit(x_train,y_train)

#plot the RandomForestClassifier’s first 5 trees 
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],feature_names= features.columns,
 class_names= ['Healthy', 'Parkinson'],filled = True,ax = axes[index])
    axes[index].set_title('Estimator: ' + str(index+1), fontsize = 11)
fig.savefig('Random Forest 5 Trees.png')


#predict the output for x_test
y_pred=model.predict(x_test)

#calculate accuracy,root mean squared error
print("Accuracy :",accuracy_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

#input data and transform into numpy array
for i in range(3):
    in_data= np.asarray(tuple(map(float,input("Enter the data:\n").rstrip().split(','))))
    #reshape and scale the input array
    in_data_re = in_data.reshape(1,-1)
    in_data_sca = scaler.transform(in_data_re)

    #print the predicted output for input array
    print("Parkinson's Disease Detected\n" if model.predict(in_data_sca)
          else "No Parkinson's Disease Detected\n")
