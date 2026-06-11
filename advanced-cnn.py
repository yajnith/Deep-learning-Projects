import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
import numpy as np
import sqlite3

# -------------------------------
# 1. LOAD CIFAR-10
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# -------------------------------
# 2. STORE IN DATABASE
# -------------------------------
conn = sqlite3.connect("cifar10.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS data (image BLOB, label INT)")

# store few samples (for demo)
for i in range(100):
    cursor.execute("INSERT INTO data VALUES (?, ?)", (x_train[i].tobytes(), int(y_train[i])))
conn.commit()

# -------------------------------
# 3. MODELS
# -------------------------------

# 🔹 LeNet
def lenet():
    model = models.Sequential([
        layers.Conv2D(6,(5,5),activation='relu',input_shape=(32,32,3)),
        layers.AveragePooling2D(),
        layers.Conv2D(16,(5,5),activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120,activation='relu'),
        layers.Dense(84,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    return model

# 🔹 AlexNet (simplified)
def alexnet():
    model = models.Sequential([
        layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10,activation='softmax')
    ])
    return model

# 🔹 VGG16
def vgg16():
    base = VGG16(weights=None, input_shape=(32,32,3), classes=10)
    return base

# 🔹 ResNet
def resnet():
    base = ResNet50(weights=None, input_shape=(32,32,3), classes=10)
    return base

# -------------------------------
# 4. TRAIN FUNCTION
# -------------------------------
def train(model, name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"\nTraining {name}...")
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)
    
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

# -------------------------------
# 5. TRAIN ALL
# -------------------------------
models_dict = {
    "LeNet": lenet(),
    "AlexNet": alexnet(),
    "VGG16": vgg16(),
    "ResNet": resnet()
}

results = {}

for name, model in models_dict.items():
    results[name] = train(model, name)

# -------------------------------
# 6. RESULTS
# -------------------------------
print("\nFinal Comparison:")
for k,v in results.items():
    print(k, ":", v)
