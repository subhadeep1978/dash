

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt 
 

def find_accelerometer_triple(csv_path):
    df = pd.read_csv(csv)
    timestamps = df.iloc[:,0].values
    payload    = df.iloc[:,1:].values.astype(float)
    n_samples, n_bytes = payload.shape

    results = []
    for i in range(n_bytes - 2):
        x = payload[:, i]
        y = payload[:, i+1]
        z = payload[:, i+2]

        # estimate offsets
        cx = np.median(x)
        cy = np.median(y)
        cz = np.median(z)

        rt = (x-cx)**2 + (y-cy)**2 + (z-cz)**2

        score = np.std(rt) / np.mean(rt)

        results.append({
            "i": i,
            "indices": (i, i+1, i+2),
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "score": score
        })

    results = sorted(results, key=lambda r: r["score"])

    print("\nTop candidates:")
    for r in results[:10]:
        print(r)

    winner = results[0]

    print("\nBest accelerometer triple:")
    print(winner["indices"])

    return winner

# =================
# RUN TEST
# =================
csv = sys.argv[1]
find_accelerometer_triple(csv)