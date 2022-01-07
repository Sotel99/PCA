import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def my_normalize(x):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

def myPCA(x,k):
    #Normalizing data
    x_new=my_normalize(x)
    #Calculating covariance matrix
    sigma = 1 / 2 * np.cov(x_new.T)
    #Calculating eigenvectors and values
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    #Sorting eigenvectors/values according to eigenvalues in descending order
    idx=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[idx]
    eigenvectors=eigenvectors[:,idx]

    return eigenvectors[:k]

#loading data
url = "http://lib.stat.cmu.edu/datasets/boston"
df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
data = np.hstack([df.values[::2, :], df.values[1::2, :2]])
output = df.values[1::2, 2]
#creating list which colours each house depending on its price
col=[]
for i in range(0, len(output)):
    if output[i]<=15:
        col.append('green')
    elif output[i]>=30:
        col.append('magenta')
    else:
        col.append('blue')

output=my_normalize(output)
#loading eigenvectors to variable
eigenvectors=myPCA(data, 8)
#calculating 1st and 2nd principal component
PC1 = data.dot(eigenvectors[0,:].T)
PC2 = data.dot(eigenvectors[1,:].T)
PC1=my_normalize(PC1)
PC2=my_normalize(PC2)
#plotting PC1 vs PC2 graph
for i in range(len(col)):
    plt.scatter(x=PC1[i], y=PC2[i], color=col[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 VS PC2')
plt.show()
''' My conclusion is that PC1 and especially PC2 are correlated
with the price of houses and that variance explained by PC1 is bigger than PC2.
'''
