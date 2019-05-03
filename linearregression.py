# linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

customers = pd.read_csv("Ecommerce Customers")

sns.jointplot('Time on Website','Length of Membership',data=customers)

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# equation 

X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers ['Yearly Amount Spent']


# test train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# model training 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# print the model coefficients 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# model validation 
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions) # scatter plot of the residuals 

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



