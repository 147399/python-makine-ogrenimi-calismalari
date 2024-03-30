import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from  sklearn.model_selection import train_test_split,cross_val_score
import sklearn.metrics as mt 
from sklearn.linear_model import LinearRegression ,Ridge

data = pd.read_csv("C:/Users/ASAF/Desktop/Ev2.csv")
veri= data.copy()

veri = veri.drop(columns="Address",axis=1)

y = veri["Price"]
X = veri.drop(columns="Price",axis=1)
"""
sabit = sm.add_constant(X)

vif = pd.DataFrame()
vif["Değişkenler"]= X.columns
vif["VIF"] = [variance_inflation_factor(sabit,i+1) for i in range(X.shape[1])]

print(vif)
"""

X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.2,random_state=42)

def caprazdog(model):
    dogruluk = cross_val_score(model,X,y,cv = 10)
    return dogruluk.mean()

def basari(gercek,tahmin):
    rmse = mt.mean_squared_error(gercek, tahmin,squared=True )
    r2 = mt.r2_score(gercek,tahmin)
    return [rmse,r2]



lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
lin_tahmin = lin_model.predict(X_test)



print(lin_model)
print(lin_model.coef_)
