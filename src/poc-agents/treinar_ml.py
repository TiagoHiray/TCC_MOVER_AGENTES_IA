from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(r"C:\Users\danie\Documents\tcc\git\TCC_MOVER_AGENTES_IA\src\data")
caminho_modelo = DATA_DIR / "modelo_especialista.joblib"

try:
    df = pd.read_csv("telemetria.csv") #csv oriundo do código atual
    print(f"Dataset carregado: {len(df)} frames.")
except FileNotFoundError:
    print("Erro: Arquivo não encontrado!")
    exit()

features = ['speed_mps', 'throttle', 'brake', 'steer', 'acc_x', 'acc_y', 'imu_gyro_z']
X = df[features].dropna()

print("Treinando o modelo (Isolation Forest)...")
# contamination=0.05 assume que ~5% dos dados são anomalias
modelo = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
modelo.fit(X)

joblib.dump(modelo, caminho_modelo)