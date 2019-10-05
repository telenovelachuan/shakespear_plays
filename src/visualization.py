import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  load processed dataset
processed_data = pd.read_csv("../data/processed/processed.csv")
MAX_ITEM_NUM = 30
print processed_data.info()

'''
Number of PlayerLines grouped by column Player
'''
play_data = processed_data.groupby('Player').count().sort_values(by='PlayerLine', ascending=False)['PlayerLine']
play_data = play_data.to_frame()
play_data['Player'] = play_data.index.tolist()
play_data.index = np.arange(0, len(play_data))
play_data = play_data[:MAX_ITEM_NUM]
play_data.columns = ['Lines', 'Player']

plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Lines', y='Player', data=play_data, order=play_data['Player'])
ax.set(xlabel='Number of PlayerLine', ylabel='Player')
ax.yaxis.set_label_coords(-0.05, 1.02)
plt.ylabel('Player', rotation=0)
plt.show()

'''
Frequency distribution of PlayerLine_length
'''
ax1 = processed_data['PL_length'].value_counts().sort_index()[:100].plot(kind='area')
ax1.set(xlabel='PlayerLine length', ylabel='Frequency')
plt.show()

'''

Frequency distribution of PlayerLine word count

'''
print processed_data['PL_w_count'].value_counts().sort_index()
ax1 = processed_data['PL_w_count'].value_counts().sort_index()[:50].plot(kind='area')
ax1.set(xlabel='PlayerLine word count', ylabel='Frequency')
ax1.set_xlim(0, 30)
plt.show()

'''
Pearson correlation
# '''
print processed_data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(decimals=2)
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 7))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(processed_data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(decimals=2),
            linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.xticks(rotation=45, fontsize=7)

plt.show()


