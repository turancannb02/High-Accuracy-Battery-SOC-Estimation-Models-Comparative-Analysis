# -*- coding: utf-8 -*-
"""Battery - ANN - 20231210.ipynb

# ANN

## 0 Degree
"""

# Gerekli kütüphaneleri yüklüyorum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/content/drive/MyDrive/BATTERY DATASET - GITHUB/TEC-reduced-model-main/tec_reduced_model')
from process_experimental_data import *

# Batarya sıcaklığı 0 derece için veri setini yüklüyorum
T = 0
crates = [0.5, 1, 2]
datasets = {}

# Farklı Crate değerleri için verileri yüklüyorum
for Crate in crates:
    datasets[Crate] = import_thermal_data(Crate, T)

# Yüklenen verileri kontrol ediyorum
for Crate, data in datasets.items():
    print(f"Crate: {Crate}")
    for cell_id, data_0degree in data.items():
        print(f"Cell ID: {cell_id}")
        display(data_0degree.head())

# Verilerin genel özetini alıyorum
data_0degree.describe()

# Veri seti hakkında bilgi alıyorum
display(data_0degree.info())

# Veri setinin boyutlarını kontrol ediyorum
display(data_0degree.shape)

# Sütun isimlerini kontrol ediyorum
print(data_0degree.columns)

# İlgili sütunları seçiyorum ve veri tiplerini float yapıyorum
selected_columns = ["Voltage [V]", "Current [A]", "AhAccu [Ah]", "WhAccu [Wh]", "Watt [W]", "Temp Cell [degC]"]
data_0degree_updated = data_0degree[selected_columns].astype(float)

# İlk birkaç satırı görüntülüyorum
display(data_0degree_updated.head())

# AhAccu [Ah] sütununu sona taşıyorum
column_to_move = 'AhAccu [Ah]'
column_series = data_0degree_updated.pop(column_to_move)
data_0degree_updated[column_to_move] = column_series

# Sona taşınan sütunu kontrol ediyorum
display(data_0degree_updated.head())

# Eğitim ve test veri seti boyutlarını belirliyorum
train_size = 0.8
test_size = 0.2
train_rows = data_0degree_updated.shape[0] * train_size
test_rows = data_0degree_updated.shape[0] * test_size
print("Train dataset has {} rows while test dataset has {} rows.".format(train_rows, test_rows))

# Veri setini eğitim ve test olarak bölüyorum
data_0degree_updated_train = data_0degree_updated[:45661]
data_0degree_updated_test = data_0degree_updated[45661:]
data_0degree_updated_train.shape, data_0degree_updated_test.shape

# Veriyi ölçeklendirmek için MinMaxScaler kullanıyorum
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_normalized = scaler.fit_transform(data_0degree_updated_train.values.astype(float))
test_normalized = scaler.transform(data_0degree_updated_test.values.astype(float))
train_normalized.shape, test_normalized.shape

# Veriyi sekanslar ve etiketler olarak hazırlayan fonksiyonu tanımlıyorum
import torch
def create_sequences_and_labels(data, lookback):
    X_list, y_list = [], []
    for i in range(0, len(data)):
        try:
            X = data[:,:5][i:i + lookback]
            y = data[:, 5][i + lookback]
            X_list.append(X)
            y_list.append(y)
        except:
            break
    return torch.tensor(X_list), torch.tensor(y_list)

# Geriye bakış süresini belirliyorum
lookback = 6
X_train, y_train = create_sequences_and_labels(train_normalized, lookback)
X_test, y_test = create_sequences_and_labels(test_normalized, lookback)

# Verilerin şekillerini kontrol ediyorum
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Veri tipi dönüşümleri yapıyorum
X_train = X_train.transpose(1,2).to(torch.float32)
X_test = X_test.transpose(1,2).to(torch.float32)
y_train = y_train.reshape((-1, 1)).to(torch.float32)
y_test = y_test.reshape((-1, 1)).to(torch.float32)

# Veri tipi dönüşümlerinin ardından verilerin şekillerini tekrar kontrol ediyorum
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Veri yükleyicileri oluşturuyorum
from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = 10, shuffle = True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size = 10, shuffle = False)

# Yapay Sinir Ağı (ANN) modelini tanımlıyorum
import torch
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(30, 40)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(40, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out

# Modeli CPU üzerinde çalıştırmak üzere tanımlıyorum
device = "cpu"
model = ANN().to(device)
print(model)

# Kayıp fonksiyonu ve optimizasyon algoritmasını tanımlıyorum
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modeli eğitiyorum
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Eğitim veri setinde tahminleri görselleştiriyorum
with torch.no_grad():
    predicted = model(X_train.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_train, label='Gerçek Değerler', color='cadetblue', linewidth=2)
plt.plot(predicted, label='Tahmin Edilen Değerler', color='sandybrown', linewidth=2)
plt.title('Eğitim Veri Setinde Şarj Durumu Tahmini', fontsize=16)
plt.xlabel('Zaman', fontsize=14)
plt.ylabel('Şarj Durumu', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

# Test veri setinde tahminleri görselleştiriyorum
with torch.no_grad():
    predicted_test = model(X_test.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Gerçek Değerler', color='cadetblue', linewidth=2)
plt.plot(predicted_test, label='Tahmin Edilen Değerler', color='sandybrown', linewidth=2)
plt.title('Test Veri Setinde Şarj Durumu Tahmini', fontsize=16, **csfont)
plt.xlabel('Zaman', fontsize=14)
plt.ylabel('SoC', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

# Eğitim ve test veri seti için metrikleri hesaplıyorum
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_train_np = y_train.numpy()
y_test_np = y_test.numpy()
predicted_train_np = predicted.reshape(-1)
predicted_test_np = predicted_test.reshape(-1)

mae_train = mean_absolute_error(y_train_np, predicted_train_np)
mse_train = mean_squared_error(y_train_np, predicted_train_np)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train_np, predicted_train_np)

print(f"Eğitim Veri Seti için Metrikler:")
print(f"MAE: {mae_train}")
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"R2: {r2_train}")

mae_test = mean_absolute_error(y_test_np, predicted_test_np)
mse_test = mean_squared_error(y_test_np, predicted_test_np)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test_np, predicted_test_np)

print(f"Test Veri Seti için Metrikler:")
print(f"MAE: {mae_test}")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"R2: {r2_test}")

# Veri analizini ve korelasyon matrisini görselleştiriyorum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.DataFrame(train_normalized, columns=["Voltage [V]", "Current [A]", "AhAccu [Ah]", "WhAccu [Wh]", "Watt [W]", "Temp Cell [degC]"])

corr_matrix = train_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Oranges', fmt=".2f", linewidths=.5)
plt.title("Korelasyon Matrisi")
plt.show()

# Modeli kaydediyorum
path = "/content/drive/MyDrive/BATTERY DATASET - GITHUB/data_0degree.pth"
torch.save(model, path)

"""## 10 Degree"""
# 10 derece sıcaklık için benzer işlemleri gerçekleştiriyorum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/content/drive/MyDrive/BATTERY DATASET - GITHUB/TEC-reduced-model-main/tec_reduced_model')
from process_experimental_data import *

T = 10
crates = [0.5, 1, 2]
datasets = {}

for Crate in crates:
    datasets[Crate] = import_thermal_data(Crate, T)

for Crate, data in datasets.items():
    print(f"Crate: {Crate}")
    for cell_id, data_10degree in data.items():
        print(f"Cell ID: {cell_id}")
        display(data_10degree.head())

display(data_10degree.info())

display(data_10degree.shape)

selected_columns = ["Voltage [V]", "Current [A]", "AhAccu [Ah]", "WhAccu [Wh]", "Watt [W]", "Temp Cell [degC]"]
data_10degree_updated = data_10degree[selected_columns].astype(float)

display(data_10degree_updated.head())

column_to_move = 'AhAccu [Ah]'
column_series = data_10degree_updated.pop(column_to_move)
data_10degree_updated[column_to_move] = column_series

display(data_10degree_updated.head())

train_size = 0.8
test_size = 0.2
train_rows = data_10degree_updated.shape[0]*train_size
test_rows = data_10degree_updated.shape[0]*test_size
print("Train dataset has {} rows while test dataset has {} rows.".format(train_rows, test_rows))

data_10degree_updated_train = data_10degree_updated[:45012]
data_10degree_updated_test = data_10degree_updated[45012:]
data_10degree_updated_train.shape, data_10degree_updated_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_normalized = scaler.fit_transform(data_10degree_updated_train.values.astype(float))
test_normalized = scaler.transform(data_10degree_updated_test.values.astype(float))
train_normalized.shape, test_normalized.shape

import torch
def create_sequences_and_labels(data, lookback):
    X_list, y_list = [], []
    for i in range(0, len(data)):
        try:
            X = data[:,:5][i:i + lookback]
            y = data[:, 5][i + lookback]
            X_list.append(X)
            y_list.append(y)
        except:
            break
    return torch.tensor(X_list), torch.tensor(y_list)

lookback = 6
X_train, y_train = create_sequences_and_labels(train_normalized, lookback)
X_test, y_test = create_sequences_and_labels(test_normalized, lookback)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train.transpose(1,2).to(torch.float32)
X_test = X_test.transpose(1,2).to(torch.float32)
y_train = y_train.reshape((-1, 1)).to(torch.float32)
y_test = y_test.reshape((-1, 1)).to(torch.float32)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = 10, shuffle = True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size = 10, shuffle = False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(30, 40)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(40, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out

device = "cpu"
model = ANN().to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

with torch.no_grad():
    predicted = model(X_train.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_train, label='Actual Values', color='blue')
plt.plot(predicted, label='Predicted Values', color='red')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.legend()
plt.show()

with torch.no_grad():
    predicted_test = model(X_test.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(predicted_test, label='Predicted Values', color='red')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.legend()
plt.show()

path = "/content/drive/MyDrive/BATTERY DATASET - GITHUB/data_10degree.pth"
torch.save(model, path)

"""## 25 Degrees"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/content/drive/MyDrive/BATTERY DATASET - GITHUB/TEC-reduced-model-main/tec_reduced_model')
from process_experimental_data import *

T = 25
crates = [0.5, 1, 2]
datasets = {}

for Crate in crates:
    datasets[Crate] = import_thermal_data(Crate, T)

for Crate, data in datasets.items():
    print(f"Crate: {Crate}")
    for cell_id, data_25degree in data.items():
        print(f"Cell ID: {cell_id}")
        display(data_25degree.head())

display(data_25degree.info())

display(data_25degree.shape)

selected_columns = ["Voltage [V]", "Current [A]", "AhAccu [Ah]", "WhAccu [Wh]", "Watt [W]", "Temp Cell [degC]"]
data_25degree_updated = data_25degree[selected_columns].astype(float)

display(data_25degree_updated.head())

column_to_move = 'AhAccu [Ah]'
column_series = data_25degree_updated.pop(column_to_move)
data_25degree_updated[column_to_move] = column_series

display(data_25degree_updated.head())

train_size = 0.8
test_size = 0.2
train_rows = data_25degree_updated.shape[0] * train_size
test_rows = data_25degree_updated.shape[0] * test_size
print("Train dataset has {} rows while test dataset has {} rows.".format(train_rows, test_rows))

data_25degree_updated_train = data_25degree_updated[:10706]
data_25degree_updated_test = data_25degree_updated[10706:]
data_25degree_updated_train.shape, data_25degree_updated_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_normalized = scaler.fit_transform(data_25degree_updated_train.values.astype(float))
test_normalized = scaler.transform(data_25degree_updated_test.values.astype(float))
train_normalized.shape, test_normalized.shape

import torch
def create_sequences_and_labels(data, lookback):
    X_list, y_list = [], []
    for i in range(0, len(data)):
        try:
            X = data[:,:5][i:i + lookback]
            y = data[:, 5][i + lookback]
            X_list.append(X)
            y_list.append(y)
        except:
            break
    return torch.tensor(X_list), torch.tensor(y_list)

lookback = 6
X_train, y_train = create_sequences_and_labels(train_normalized, lookback)
X_test, y_test = create_sequences_and_labels(test_normalized, lookback)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train.transpose(1,2).to(torch.float32)
X_test = X_test.transpose(1,2).to(torch.float32)
y_train = y_train.reshape((-1, 1)).to(torch.float32)
y_test = y_test.reshape((-1, 1)).to(torch.float32)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = 5, shuffle = True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size = 5, shuffle = False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(30, 40)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(40, 50)
        self.layer3 = nn.Linear(50, 60)
        self.layer4 = nn.Linear(60, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        out = self.layer4(x)
        return out

device = "cpu"
model = ANN().to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

with torch.no_grad():
    predicted = model(X_train.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_train, label='Actual Values', color='blue')
plt.plot(predicted, label='Predicted Values', color='red')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.legend()
plt.show()

with torch.no_grad():
    predicted_test = model(X_test.to(device)).to(device).numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(predicted_test, label='Predicted Values', color='red')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.legend()
plt.show()

path = "/content/drive/MyDrive/BATTERY DATASET - GITHUB/data_25degree.pth"
torch.save(model, path)

