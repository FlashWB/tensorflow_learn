import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y) #类似mnist 变成10个数，1代表0-9中数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    










