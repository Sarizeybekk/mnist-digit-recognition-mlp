
from tensorflow.keras.datasets import mnist # hazır rakam veriseti
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
import tensorflow as tf
import numpy as np 

(x_train,y_train),(x_test,y_test) = mnist.load_data()

def img_show_save():
    img =x_train[0]
    label = y_train[0]

    plt.imsave("a.png",img,cmap ="gray")
    plt.imshow(img,cmap= "gray")
    plt.show()
    print(f"{label} numarası img olrak kaydedildi")

print(x_train.shape)
x_train =x_train.reshape(-1,28*28)/255.0 #flattening
x_test =x_test.reshape(-1,28*28)/255.0

# 2x2 lik bir resim [5,10
#                  15,20]
#[5,10,15,20]

model = models.Sequential([
    tf.keras.Input(shape=(784,)), #girdi verisi 784 boyutunda (28*28)
    layers.Dense(64,activation="relu"), # gizli katman--> nöron sayısı  activation noronların aralarındaki konuşma gibi düşün bunların genel katmanını ayarlayan relu,softmax,
    layers.Dense(10,activation="softmax") # çıkış katmanı --> 10 olasılık var...


    ])
#model compile
# epoch
#optimizer optimizasyon yapıcak
#loss kayıp fonksiyonu
model.compile(optimizer ="adam",loss ="sparse_categorical_crossentropy", metrics =["accuracy"])
#Adaptive Moment Estimation
#0-9 arası sayısal  etiketler için
# kaç kere veriyi sıfırdan alarak egitsin epochs
# batch_size default her turdaki verinin oransal olraka kaçını alacagım.
# iterasyon---> her batchin yaptıgı bir adımı temsil eder 
# epoch=5,iteration=938,batch_size=64
model.fit(x_train,y_train,epochs=5,batch_size=64)


test_loss,test_acc =model.evaluate(x_test,y_test)# predict 
print(f"Test dogruluk oranı {test_acc}")

# rastgele test ornegi 
index= np.random.randint(0,len(x_test))
sample =x_test[index].reshape(1,28*28)



prediction= model.predict(sample) 
predictied_label = np.argmax(prediction) #argmax=> en yüksek olasılık olarak al
print(predictied_label)

plt.imshow(x_test[index].reshape(28,28),cmap="gray")
plt.title(f"Tahmin edilen :{predictied_label}  Gerçek :{y_test[index]}")
plt.axis('off')
plt.show()

model.save("model.h5") #.keras uzantısıda mümkün 



