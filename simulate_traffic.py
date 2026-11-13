import json
import sys
import urllib.request

import pandas as pd

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
df = pd.read_csv("power_plant.csv").sample(200, random_state=1)
if mode == "drift":
    df["AT"] = df["AT"] + 8      # pretend it got 8 degrees hotter

for _, row in df.iterrows():
    body = json.dumps({c: float(row[c]) for c in ["AT", "V", "AP", "RH"]}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict", data=body,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()

print(f"Sent 200 '{mode}' requests")