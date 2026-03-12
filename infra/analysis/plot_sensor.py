import sys
import pandas as pd
import matplotlib.pyplot as plt 
 

csv = sys.argv[1]
df  = pd.read_csv(csv) 
print(df.columns)


f = plt.figure(figsize=(15,15))
ncol = df.shape[1]-1

for c in range(ncol):
    ax = f.add_subplot(ncol, 1, c+1)
    ax.plot(df[ str(c) ], color="red")
    ax.set_ylim([0,255])
    ax.set_ylabel(c)
    ax.set_xticks([])

plt.tight_layout()
plt.show()





