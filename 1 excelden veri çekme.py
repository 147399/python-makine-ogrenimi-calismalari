import pandas as pd
from sklearn.model_selection import train_test_split
 
# Excel dosyasını okuyarak veri setini yükleyin
veri = pd.read_excel("C:/Users/ASAF/Desktop/Kitap1.xlsx")

# Veri setini yazdırın
print(veri)

# Bağımlı değişkeni (etiket) 'Y' olarak ayarlayın
y = veri["Y"]

# Bağımsız değişkenleri (özellikler) 'X1' ve 'X2' olarak ayarlayın
X = veri[["X1", "X2"]]

# Veri setini eğitim ve test setlerine bölmek için train_test_split metodunu kullanın
# test_size=0.2, eğitim için %80, test için %20 oranında ayarlama yapar
# random_state=42, veri setinin rastgele bölünmesi için kullanılan başlangıç durumu (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim setini yazdırın
print(X_train)

# Test setini yazdırın
print(X_test)

# Eğitim setindeki tüm değerlerin toplamını yazdırın
print(X_train.sum())