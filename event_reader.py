#event_reader.py

import pandas as pd
ev = pd.read_csv("dataset/run_XYZ-colisao/eventos.csv")
print(ev["tipo"].value_counts())