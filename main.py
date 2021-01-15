import pandas as pd
import matplotlib.pyplot as plt

courseDF = pd.DataFrame(pd.read_excel('RC_F01_01_2017_T01_01_2020.xlsx'))
print(courseDF.head())
print(courseDF.describe())
print(courseDF.info())
courseDF['curs'].hist()
plt.show()
