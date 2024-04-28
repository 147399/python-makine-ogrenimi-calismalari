import pandas as pd

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
"""
karar agacı yapısı makıne ezberıne gıttıgı ıcın daha dogru bır tahmın hyapabılmek ıcın karar ormanı yapısı algorıtmasını kullandık
"""
# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/maas.csv")

# Bağımlı ve bağımsız değişkenleri ayır
X = data["YearsExperience"].values.reshape(-1, 1)  # Bağımsız değişken
y = data["Salary"].values.reshape(-1, 1)           # Bağımlı değişken

# RandomForestRegressor modelini oluştur ve eğit
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X, y)

# DecisionTreeRegressor modelini oluştur ve eğit
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X, y)

# Tahminleri yap
rf_tahmin = rf_model.predict(X)
dt_tahmin = dt_model.predict(X)

# Sonuçları görselleştir
plt.scatter(X, y, color="red", label="Gerçek Veri")
plt.plot(X, rf_tahmin, color="blue", label="Random Forest Tahmini")
plt.plot(X, dt_tahmin, color="green", label="Decision Tree Tahmini")
plt.xlabel("Yıl Deneyimi")
plt.ylabel("Maaş")
plt.legend()
plt.show()
