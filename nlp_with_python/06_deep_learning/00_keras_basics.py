import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_categorical

iris = load_iris()
print(iris.DESCR)

X = iris.data
y = iris.target
