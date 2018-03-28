import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use("ggplot")


data = pd.read_csv('input/pokemon.csv')
#print(data.head())

# tail shows last 5 rows
#print(data.tail())

#numbers of row and columns
#print(data.shape)

# For example lets look frequency of pokemom types
#print(data['Type 1'].value_counts(dropna =True))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon

# For example max HP is 255 or min defense is 5
#print(data.dtypes)
#print(data.describe()) #ignore null entries

#For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack',by = 'Legendary')
plt.show()