import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import fetch_openml

# Ames konut veri setini yükle
housing = fetch_openml(name="house_prices", as_frame=True)

# Veri ve hedef değişkenlerini al
data = housing.data
target = housing.target

# Veriyi DataFrame'e dönüştür
veri = data.copy()
veri["PRICE"] = target

# Bağımsız ve bağımlı değişkenleri ayır
X = veri.drop(columns=["PRICE"])
y = veri["PRICE"]
X = pd.get_dummies(X)
X_train , X_test , y_train , y_test =train_test_split(X,y,test_size=0.1 ,random_state=42)    

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)


print(ridge_model.coef_)
print(lasso_model.coef_)

