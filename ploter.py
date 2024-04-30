import pandas as pd
from matplotlib import pyplot as plt

d0 = pd.read_csv('0.csv')
d1 = pd.read_csv('1.csv')
d2 = pd.read_csv('2.csv')
d3 = pd.read_csv('3.csv')
d4 = pd.read_csv('4.csv')
d5 = pd.read_csv('5.csv')

data = [d0.to_numpy()[:, 1], d1.to_numpy()[:, 1], d2.to_numpy()[:, 1],
        d3.to_numpy()[:, 1], d4.to_numpy()[:, 1], d5.to_numpy()[:, 1]]


for i in range(len(data)):
    data[i] = data[i] * 1000


plt.boxplot(data[1:], showfliers=False, vert=False)
 # https://rowannicholls.github.io/python/graphs/ax_based/boxplots_significance.html use to make statment about sigifcance
# plt.yticks([1,2,3,4,5, 6], ['0', '1', '2', '3', '4', '5'])

plt.title('Effects of opening and closing operations on performance')
plt.xlabel('Milliseconds per Frame')
plt.ylabel('Opening and Closing Count')


plt.savefig('eff2.png')
