import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("Task 1A\\Data\\train.csv")
    #y = data["y"].to_numpy()
    #data = data.drop(columns="y")
    # print a few data samples
    plot = sns.pairplot(data)
    plt.savefig("Task 1A\\seaborn_plot.png")
