import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


new_cases = pd.read_csv("data/new_cases.csv")
new_deaths = pd.read_csv("data/new_deaths.csv")
total_cases = pd.read_csv("data/total_cases.csv")
total_deaths = pd.read_csv("data/total_deaths.csv")

print(new_cases["France"].head())
print(new_cases["France"].describe())
"""print(new_deaths["France"].describe())
print(total_cases["France"].describe())
print(total_deaths["France"].describe())"""

def make_plot(axs, data, country):
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    ax1 = axs[0, 0]
    ax1.plot(data[0])
    ax1.set_title('new_cases ' + country)
    #ax1.set_ylabel('misaligned 1', bbox=box)
    ax1.set_ylim(0, 2000)

    ax3 = axs[1, 0]
    ax3.set_title('new_deaths ' + country)
    #ax3.set_ylabel('misaligned 2', bbox=box)
    ax3.plot(data[1])

    ax2 = axs[0, 1]
    ax2.set_title('total_cases ' + country)
    ax2.plot(data[2])
    #ax2.set_ylabel('aligned 1', bbox=box)
    ax2.set_ylim(0, 2000)

    ax4 = axs[1, 1]
    ax4.plot(data[3])
    #ax4.set_ylabel('aligned 2', bbox=box)
    ax4.set_title('total_deaths ' + country)

# Plot 1:
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(left=0.2, wspace=0.6)
data = [new_cases["France"], new_deaths["France"], total_cases["France"], total_deaths["France"]]
make_plot(axs, data, "France")

# just align the last column of axes:
fig.align_ylabels(axs[:, 1])
#plt.show()