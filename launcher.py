# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use("ggplot")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "input"]).decode("utf8"))

#print(os.listdir('input'))

data = pd.read_csv('input/pokemon.csv')
#data.info() #---> Get information on frame

#correlation map
#f,ax = plt.subplots(figsize=(18, 18))
#sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#plt.show()

#print(data.head(10)) # Get the ten first value

#print(data.dtypes)