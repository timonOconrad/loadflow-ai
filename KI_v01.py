import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Pfad zur YAML-Datei
yaml_file_path = 'config/default.yaml'

with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Pfad zur Parquet-Datei
parquet_file_path = config['data']['parquet_file_path']





# Parquet-Datei einlesen
df = pd.read_parquet(parquet_file_path)

# DataFrame anzeigen
print(df)

column_names = df.columns
print(column_names)



# Teilen Sie die Spalten mit Arrays in separate Spalten auf
df[['P_G_1', 'P_G_2', 'P_G_3', 'P_G_4', 'P_G_5']] = pd.DataFrame(df['P_G'].tolist(), index=df.index)
df[['Q_G_1', 'Q_G_2', 'Q_G_3', 'Q_G_4', 'Q_G_5']] = pd.DataFrame(df['Q_G'].tolist(), index=df.index)
df[['P_L_1', 'P_L_2', 'P_L_3', 'P_L_4', 'P_L_5']] = pd.DataFrame(df['P_L'].tolist(), index=df.index)
df[['Q_L_1', 'Q_L_2', 'Q_L_3', 'Q_L_4', 'Q_L_5']] = pd.DataFrame(df['Q_L'].tolist(), index=df.index)

df[['u_1_real', 'u_2_real', 'u_3_real', 'u_4_real', 'u_5_real']] = pd.DataFrame(df['u_powerfactory_real'].tolist(), index=df.index)
df[['u_1_imag', 'u_2_imag', 'u_3_imag', 'u_4_imag', 'u_5_imag']] = pd.DataFrame(df['u_powerfactory_imag'].tolist(), index=df.index)
df[['evaluation']] = pd.DataFrame(df['evaluation_new'].tolist(), index=df.index)

def berechne_P_Q(df):
    for i in range(1, 6):
        df[f'P_{i}'] = df[f'P_G_{i}'] + df[f'P_L_{i}']
        df[f'Q_{i}'] = df[f'Q_G_{i}'] + df[f'Q_L_{i}']
    return df

df = berechne_P_Q(df)
def berechne_U_abs(df):
    for i in range(1, 6):
        df[f'U_{i}'] = np.sqrt(df[f'u_{i}_real']**2 + df[f'u_{i}_imag']**2)
    return df

df = berechne_U_abs(df)

# Wählen Sie die relevanten Spalten für die Transformation aus
X = df[[ 'u_1_real', 'u_2_real','U_1','U_2','P_2', 'P_3', 'P_4', 'P_5',
        'Q_3', 'u_1_imag', 'Q_4', 'Q_5']]
Y = df[['P_1','Q_1', 'Q_2', 'u_2_real', 'u_3_real', 'u_4_real', 'u_5_real', 'u_2_imag', 'u_3_imag', 'u_4_imag', 'u_5_imag', 'U_3', 'U_4', 'U_5']]


# Normalisieren Sie die Daten mit StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)


trainingsdaten_groesse = X.shape[0]
print("Datensatz Größe :", trainingsdaten_groesse)



# Annehmen, dass X und Y Ihre ursprünglichen Daten sind
X, Y = shuffle(X, Y, random_state=42)  # Daten mischen

# Anzahl der Trainingsdaten definieren
train_size = config['data']['train_size']

if train_size * 1.3 >= trainingsdaten_groesse:
    train_size = int(trainingsdaten_groesse /1.3)

# Aufteilen der Daten
X_train = X[:train_size]
Y_train = Y[:train_size]

# Der Rest der Daten wird aufgeteilt in Validierung und Test
X_temp = X[train_size:]
Y_temp = Y[train_size:]

# Halbieren der verbleibenden Daten für Validierung und Test
test_val_size = int(train_size * 0.15)
X_val = X_temp[:test_val_size]
Y_val = Y_temp[:test_val_size]
X_test = X_temp[test_val_size:]
Y_test = Y_temp[test_val_size:]

# Konvertieren in PyTorch-Tensoren
X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)
X_val = torch.Tensor(X_val)
Y_val = torch.Tensor(Y_val)
X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)



# Ausgabe der Datensatzgrößen
print("Trainingsdaten Größe:", X_train.size(0))
print("Validierungsdaten Größe:", X_val.size(0))
print("Testdaten Größe:", X_test.size(0))

# Definition eines einfachen neuronalen Netzes mit PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 14)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modellinstanz erstellen
model = Net()

# Verlustfunktion und Optimierer definieren
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam Optimizer
date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# CSV-Datei für das Protokoll vorbereiten
csv_file = f"train/training_progress_{date_time}.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Size', 'Validation Size', 'Timestamp'])

# Training des Modells
epochs = config['params-KI']['epochs']

for epoch in range(epochs):
    # Trainingsphase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    # Validierungsphase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, Y_val)

    # Fortschritt in CSV-Datei speichern
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss.item(), val_loss.item(), X_train.size(0), X_val.size(0), datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    # Ausgabe von Trainings- und Validierungsverlust
    if (epoch + 1) % 10 == 0:  # Zum Beispiel alle 10 Epochen
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')


# Validierung des Modells
model.eval()
with torch.no_grad():
    val_predictions = model(X_val)

# Berechnen und Anzeigen der Validierungsgenauigkeit (z.B. Mean Squared Error)
val_mse = mean_squared_error(Y_val, val_predictions)
print(f'Validation Mean Squared Error: {val_mse:.8f}')

# Testen des Modells
with torch.no_grad():
    test_predictions = model(X_test)

# Berechnen und Anzeigen der Testgenauigkeit (z.B. Mean Squared Error)
test_mse = mean_squared_error(Y_test, test_predictions)
print(f'Test Mean Squared Error: {test_mse:.8f}')


torch.save(model, f"train/model_{date_time}.pth")


# Gewichte laden
model_loaded = torch.load(f"train/model_{date_time}.pth")
model_loaded.eval()

with torch.no_grad():
    test_predictions_loaded = model_loaded(X_test)

# Berechnen und Anzeigen der Testgenauigkeit (z.B. Mean Squared Error)
test_mse = mean_squared_error(Y_test, test_predictions_loaded)
print(f'Test Mean Squared Error Loaded: {test_mse:.8f}')


# Wählen Sie einen zufälligen Fall aus dem Testdatensatz
random_sample_index = np.random.randint(0, len(X_test))
random_sample_X = X_test[random_sample_index]
random_sample_Y_true = Y_test[random_sample_index]
random_sample_Y_pred = test_predictions[random_sample_index]

# Konvertieren Sie die Tensorwerte in NumPy-Arrays für die Grafik
random_sample_X = random_sample_X.numpy()
random_sample_Y_true = random_sample_Y_true.numpy()
random_sample_Y_pred = random_sample_Y_pred.numpy()



# Erstellen und Anzeigen einer Grafik für die Abweichung der Werte für den zufälligen Fall
plt.figure(figsize=(10, 6))
plt.plot(random_sample_Y_true, label='True Values', marker='o')
plt.plot(random_sample_Y_pred, label='Predicted Values', marker='x')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.title('True vs. Predicted Values for a Random Sample')
plt.legend()
plt.grid(True)
plt.show()

# Umkehrtransformation der vorhergesagten Werte in den ursprünglichen Bereich
random_sample_Y_pred_original_scale = scaler.inverse_transform(random_sample_Y_pred.reshape(1, -1))
random_sample_Y_true_original_scale = scaler.inverse_transform(random_sample_Y_true.reshape(1, -1))

# Ausgabe der vorhergesagten Werte im ursprünglichen Bereich
print("Predicted Values (Original Scale):", random_sample_Y_pred_original_scale)
print("True Values (Original Scale):", random_sample_Y_true_original_scale)

