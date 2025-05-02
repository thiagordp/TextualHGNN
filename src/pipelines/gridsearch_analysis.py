import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns  # for prettier plots, optional

RELEVANT_COLUMNS = ["LR", "INNER_DIM", "BATCH_SIZE", "SOFTMAX_ASSIGN", "DECREASE_PROPORTION"]

DATASET = "IMDB"
df = pd.read_excel(f'gridsearch_results_{DATASET}.xlsx')


sns.violinplot(x='LR', y='VAL_MACRO_F1', data=df)

plt.title('Macro F1 Distribution per Learning Rate (LR)')
plt.show()

g = sns.FacetGrid(df, col='BATCH_SIZE', col_wrap=3, height=4, sharey=True)
g.map(sns.violinplot, 'LR', 'VAL_MACRO_F1', order=sorted(df['LR'].unique()))
g.fig.suptitle('Macro F1 vs LR across BATCH_SIZE', y=1.02)
plt.show()

pivot = df.pivot_table(values='VAL_MACRO_F1', index='LR', columns='BATCH_SIZE', aggfunc='mean')

sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title('Mean Macro F1 for LR vs BATCH_SIZE')
plt.show()

from itertools import combinations
"""
for param1, param2 in combinations(RELEVANT_COLUMNS, 2):
    pivot = df.pivot_table(values='VAL_MACRO_F1', index=param1, columns=param2, aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f'Mean Macro F1: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()
"""

import scipy.stats as stats

def summarize_param_effect(df, param):
    summary = df.groupby(param)['VAL_MACRO_F1'].agg(['mean', 'count', 'std'])
    summary['sem'] = summary['std'] / summary['count']**0.5
    summary['ci95'] = summary['sem'] * stats.t.ppf((1 + 0.95) / 2, summary['count'] - 1)
    summary['ci_lower'] = summary['mean'] - summary['ci95']
    summary['ci_upper'] = summary['mean'] + summary['ci95']
    summary = summary.sort_values(by='mean', ascending=False)
    return summary

for param in RELEVANT_COLUMNS:
    summary = summarize_param_effect(df, param)
    print(f"\nParam: {param}")
    print(summary[['mean', 'ci_lower', 'ci_upper']])
    print(f"Best value for {param}: {summary.index[0]} (Mean Macro F1: {summary.iloc[0]['mean']:.4f})")
