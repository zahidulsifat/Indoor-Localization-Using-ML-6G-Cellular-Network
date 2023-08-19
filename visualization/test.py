import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = pd.read_csv("medicoes.csv")

sns.pairplot(data=DATA)

plt.show()