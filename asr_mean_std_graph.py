import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

name = 'asr_vicuna_values_dan'
title_name = 'ASR on vicuna-7b with DAN attack and Long-Context Defense (values)'

data = pd.read_csv('/Users/joycequ/Documents/UROP/Context/llm-context/asr_results/' + name + '.csv')
x = 'Long-Context Length'
y = 'ASR'

# Calculate the mean and standard deviation for each x value
mean_df = data.groupby(x).agg({y: 'mean'}).reset_index()
std_df = data.groupby(x).agg({y: 'std'}).reset_index()
mean_df['std'] = std_df[y]

# Plot the mean with standard deviation as a shaded area
plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_df, x=x, y=y, label='Mean')
plt.fill_between(mean_df[x], 
                 mean_df[y] - mean_df['std'], 
                 mean_df[y] + mean_df['std'], 
                 color='b', alpha=0.2, label='Standard Deviation')

plt.xlabel(x)
plt.xticks(np.arange(0, 11000, 1000))
plt.ylabel(y)

plt.title(title_name)
plt.legend()


plt.ioff()
plt.savefig('/Users/joycequ/Documents/UROP/Context/llm-context/asr_results/' + name + '.png', dpi=300)
plt.show()