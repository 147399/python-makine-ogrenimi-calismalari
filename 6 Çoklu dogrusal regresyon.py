import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veri setini oku ve kopyala
data = pd.read_csv("C:/Users/ASAF/Desktop/Reklam.csv")
veri = data.copy()

# "Newspaper" sütununda aykırı değerleri belirle
Q1 = veri["Newspaper"].quantile(0.25)
Q3 = veri["Newspaper"].quantile(0.75)
IQR = Q3 - Q1
ustsınır =  Q3 + 1.5 * IQR
aykırı = veri["Newspaper"] > ustsınır
# Aykırı değerleri üst sınır ile değiştir
veri.loc[aykırı, "Newspaper"] = ustsınır

# Bağımlı değişkeni (Sales) ve bağımsız değişkenleri (TV, Radio, Newspaper) belirle
y = veri["Sales"]
X = veri[["TV", "Radio", "Newspaper"]]

# Sabit terimi ekleyerek doğrusal regresyon modelini oluştur
sabit = sm.add_constant(X)
model = sm.OLS(y, sabit).fit()
# Modelin özetini yazdır
print(model.summary())

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Eğitim seti üzerinde doğrusal regresyon modelini uygula
lr = LinearRegression()
lr.fit(X_train, y_train)
# Modelin katsayılarını yazdır
print(lr.coef_)

# Test seti üzerinde tahmin yap
tahmin = lr.predict(X_test)
# Gerçek ve tahmin edilen değerleri sıralı şekilde yazdır
y_test = y_test.sort_index()
print(tahmin)

# Gerçek ve tahmin edilen değerleri içeren bir veri çerçevesi oluştur
df = pd.DataFrame({"Gercek": y_test, "Tahmin": tahmin})
# Gerçek ve tahmin edilen değerleri içeren bir çizgi grafiği çiz
df.plot(kind="line")
plt.show()

"""
sns.pairplot(veri, kind="reg")  #dagılım grafigi olusturmak icin  kullanılır
sns.boxplot(veri["Newspaper"]) #kutu grafigi cizer
plt.show()
print(veri.isnull().sum()) #verideki boş satırlara bakar ve toplamını yazdırır
print(veri.dtypes) #verinin türüne bakar
print(veri.corr()["Sales"]) # aralarındaki ilişkiye bakar

"""