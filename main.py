import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Data Collection
df = pd.read_csv('hearts.csv')
print(df)

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder # To change the categorical data
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])# To change the categorical data into numerical and save in the same column
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])

print(df)#(Naming the categorical data as 0,1,2,3,4 as per Alphabetical order)

#Data Splitting
X = df.drop(columns=['HeartDisease'])# All the columns except HeartDisease is Independent Variable or Input (Dropping the Heart Disease Column insted of calling all the column names )
Y = df['HeartDisease']# The last column HeartDisease is Dependent Variable or Output(Taking only the HeartDisease Column)

print('Input ',X)
print('Output ',Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.2,random_state=12) #Split the data into 80% training and 20% testing(0.2 if we give the 0.3 then it will be 70% training and 30% testing) random is used to train the data in shuffeled manner 
#Check the data splitting
print("Total data",df.shape)
print(" Input Training Data",x_train.shape)
print("Input testing Data",x_test.shape)
print("Output Training Data",y_train.shape)
print("Output testing Data",y_test.shape)

#Model Building
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train,y_train)

#Model Evaluation
y_pred = NB.predict(x_test)
print('Y predict =',y_pred)
print('Y test =',y_test)

#Test the Accuracy 
from sklearn.metrics import accuracy_score
print("Accuracy Score = ",accuracy_score(y_test,y_pred)*100 ,"%")

#Model Prediction
test_prediction = NB.predict([[60,1,3,145,233,1,0,150,0,2.3,0]])
if test_prediction == 1:
    print("The Person has Heart Disease ,Please conduct to the doctor")
else:
    print("The Person has no Heart Disease")
# import pickle
# pickle.dump(NB,open('model.pkl','wb'))

