# importing package
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import statistics

parser = argparse.ArgumentParser(description='Plot learning curve')
parser.add_argument('--file', '-f', type=str, help='cfile to plot')
parser.add_argument('--factor', default='all', type=str, help="choose factor to plot")
args = parser.parse_args()

df = pd.read_csv('results/' + args.file)
i = np.arange(len(df['epoch']))

print(df)
if 'val_metrics' in df.keys():
    val_metrics = list(df['val_metrics'])
    max = max(val_metrics)
    print(max)
    print(val_metrics.index(max))
    print(val_metrics[-1])
    print(statistics.mean(val_metrics[19:]))
for k in df.keys()[2:]:    
    if (args.factor in k) or (args.factor == 'all'):
        plt.plot(i, df[k], label=k)

plt.legend()
plt.show()