# scikit-learn --> ML Kütüphanesi
# pyTorch tensorflow -> DL Kütüphanesi 

from tensorflow.keras.datasets import mnist # hazır rakam veriseti
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
import tensorflow as tf
(x_train,y_train),(x_test,y_test) = mnist.load_data()

def img_show_save():
    img =x_train[0]
    label = y_train[0]

    plt.imsave("a.png",img,cmap ="gray")
    plt.imshow(img,cmap= "gray")
    plt.show()
    print(f"{label} numarası img olrak kaydedildi")

print(x_train.shape)
x_train =x_train.reshape(-1,28*28)/255.0 #flatting
x_test =x_test.reshape(-1,28*28)/255.0

# 2x2 lik bir resim [5,10
#                  15,20]
#[5,10,15,20]

model = models.Sequential([
    tf.keras.Input(shape=(784,)), #girdi verisi 784 boyutunda (28*28)
    layers.Dense(64,activation="relu"), # gizli katman--> nöron sayısı  activation noronların aralarındaki konuşma gibi düşün bunların genel katmanını ayarlayan relu,softmax,
    layers.Dense(10,activation="softmax") # çıkış katmanı --> 10 olasılık var...


    ])
print(x_train.shape)



