#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#SET STYLE FOR VISUALS
sns.set_style('whitegrid')

#READ FROM CSV
ad_data = pd.read_csv('C:\\Users\\User\\Documents\\advertising.csv')

#Change Working Directory if required
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\LogisticRegression')

#DATAFRAME HEAD
print('\n',ad_data.head(),'\n')

#DATAFRAME INFO
print('\n',ad_data.info())

#DATAFRAME STATISTICS
print('\n',ad_data.describe())

#AGE DISTRIBUTION
i1 = sns.displot(data=ad_data,x='Age',bins=40)
i1.savefig('Age Distribution.jpg')
plt.show()

#AGE VS AVG AREA INCOME
i2 = sns.jointplot(data=ad_data,x='Age',y='Area Income',xlim=(10,70),ylim=(10000,90000))
i2.savefig('Age vs Area Income.jpg')
plt.show()

#AGE VS DAILY TIME SPENT ON SITE KDE PLOT
i3 = sns.jointplot(data=ad_data,x='Age',y='Daily Time Spent on Site',ylim=(20,100),kind='kde',color='red',fill=True)
i3.savefig('Age vs Daily Time Spent on Site.jpg')
plt.show()

#DAILY TIME SPENT ON SITE VS DAILY INTERNET USAGE
i4 = sns.jointplot(data=ad_data,x='Daily Time Spent on Site',y='Daily Internet Usage',xlim=(20,100),ylim=(50,300),color='green')
i4.savefig('Daily Time Spent on Site vs Daily Internet Usage.jpg')
plt.show()

#PAIRPLOT
i5 = sns.pairplot(data=ad_data,diag_kind='hist',hue='Clicked on Ad',palette='bwr')
i5.savefig('Pairplot.jpg')
plt.show()

#SPLITTING DATAFRAME INTO INPUT AND OUTPUT VARIABLES
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']

#TRAIN & TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=None)

#IMPORT AND FIT MODEL
from sklearn.linear_model import LogisticRegression
LogRes = LogisticRegression(max_iter=1000)
LogRes.fit(X_train,y_train)

#PREDICTIONS AND EVALUATIONS
#Coefficients
Coefficients = pd.DataFrame(data=LogRes.coef_,index=['Coefficients'],columns=[X.columns])
print('\n',Coefficients,'\n')

#Predictions
predictions = LogRes.predict(X_test)

#Classification Report
from sklearn.metrics import classification_report
print('Classification Report: \n',classification_report(y_test,predictions))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

print('Confusion Matrix: \n',confusion_matrix(y_test,predictions),'\n')

sns.set_style('white')
i6 = plot_confusion_matrix(LogRes,X_test,y_test)
plt.savefig('Confusion Matrix.jpg')
plt.show()

#END




