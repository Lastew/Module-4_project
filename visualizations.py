"""
This module is for your final visualization code.
One visualization per hypothesis question is required.
A framework for each type of visualization is provided.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set specific parameters for the visualizations
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
# plt.style.use('seaborn-whitegrid')
# sns.set_style("white")

def get_scatter_GRE_TOEFL_GCPA(df,
                               features=['GRE_Score','TOEFL_Score','CGPA'], 
                               savefig=True,save_path='img/scatter_GRE_TOEFL_GCPA',
                               context = "poster",
                               
                              ):
    """
    plot a scatter plot of GRE against TEOFL with GCPA as color
    """

    length = 20
    height = length *1/3
    dots = length * 2
    fig = plt.figure(figsize=(length,height))
    sns.set_context(context)
    sns.scatterplot(df[features[0]], df[features[1]],df[features[2]], palette='viridis', sizes=(dots,dots))
    if savefig:
        plt.savefig(save_path)
    plt.show()
    return fig
    
