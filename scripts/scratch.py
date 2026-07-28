import pandas as pd

df_raw = pd.read_stata('data/HCMST.dta', convert_categoricals=True)

# Search for matching column names in your dataset
print("Age columns:", [c for c in df_raw.columns if 'age' in c.lower() or 'page' in c.lower()])
print("Duration columns:", [c for c in df_raw.columns if 'duration' in c.lower()])
print("Breakup columns:", [c for c in df_raw.columns if 'breakup' in c.lower() or 'q21e' in c.lower()])