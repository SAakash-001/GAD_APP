import pandas as pd
import os

CSV_PATH = 'data/cv-valid-train.csv'
AUDIO_DIR = 'data/cv-valid-train'

df = pd.read_csv(CSV_PATH)
print(df.columns)  # Check actual column names!

