import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

#Dataset
Data = np.load('mnist.npz')
TranImg, TrainLbl = Data['x_train'], Data['y_train']
TestImg, TestLbl = Data['x_test'], Data['y_test']

#Preporocess
TranImg = TranImg.reshape((-1, 28, 28, 1)).astype('float32') / 255
TestImg = TestImg.reshape((-1, 28, 28, 1)).astype('float32') / 255

#CNN
Model = models.Sequential()
Model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
Model.add(layers.MaxPooling2D((2, 2)))
Model.add(layers.Conv2D(64, (3, 3), activation='relu'))
Model.add(layers.MaxPooling2D((2, 2)))
Model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Layer
Model.add(layers.Flatten())
Model.add(layers.Dense(64, activation='relu'))
Model.add(layers.Dense(10, activation='softmax'))

#Compile
Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
Model.summary()

#Train
print("\nآموزش مدل ")
History = Model.fit(TranImg, TrainLbl, epochs=5, validation_data=(TestImg, TestLbl))

#Accuracy
Loss, Acc = Model.evaluate(TestImg, TestLbl)
print("\nدقت تست: ", Acc)

#Show
plt.plot(History.history['Accuracy'], label='دقت آموزش')
plt.plot(History.history['val_accuracy'], label='دقت اعتبارسنجی')
plt.xlabel('تعداد ایپاک‌ها')
plt.ylabel('دقت')
plt.legend(loc='lower right')
plt.title('دقت مدل')
plt.show()
plt.plot(History.history['loss'], label='خطای آموزش')
plt.plot(History.history['val_loss'], label='خطای اعتبارسنجی')
plt.xlabel('تعداد ایپاک‌ها')
plt.ylabel('خطا')
plt.legend(loc='upper right')
plt.title('خطای آموزش و اعتبارسنجی مدل')
plt.show()
