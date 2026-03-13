import sys
import pandas as pd
import matplotlib.pyplot as plt 
 

# =====================================
# Command line args
# =====================================
def parseArgs():
        HELP=""
        import argparse
        import textwrap
        parser = argparse.ArgumentParser(
            formatter_class = argparse.RawTextHelpFormatter,
            description=textwrap.dedent(HELP)
        )
        parser.add_argument(
            "--sensorlog", type=str, help="Raw sensor log", required=True
        )

        parser.add_argument('--bytes', type=int, nargs='+', required=False, default=None,   help='Space separated array of bytes')


        args = parser.parse_args()
        return args

args = parseArgs()


# ====================================
csv = args.sensorlog 
df  = pd.read_csv(csv) 
print(df.columns)

colrange = args.bytes if args.bytes is not None else range(df.shape[1]-1)
ncol = len(colrange)
    

f = plt.figure(figsize=(15,15))

for idx, c in enumerate(colrange):
    ax = f.add_subplot(ncol, 1, idx+1)
    ax.step(df["time"], df[ str(c) ], color="red")
    ax.set_ylim([0,255])
    ax.set_ylabel(c)
    ax.set_xticks([])

    if idx==0: ax.set_title(csv)

plt.tight_layout()
plt.show()





