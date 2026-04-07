import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD DATASET (CIFAR-10)
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# -------------------------------
# 2. MODEL DEFINITIONS
# -------------------------------

# 1. Basic CNN
def basic_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 2. Pooling CNN
def pooling_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.AveragePooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 3. Efficient CNN (Depthwise Separable)
def efficient_cnn():
    model = models.Sequential([
        layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.SeparableConv2D(64, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 4. Dilated CNN
def dilated_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), dilation_rate=2, activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), dilation_rate=2, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 5. Random Feature CNN
def random_feature_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', trainable=False, input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', trainable=False),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# -------------------------------
# 3. TRAINING FUNCTION
# -------------------------------
def train_model(model, name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\nTraining {name} model...")
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(x_test, y_test),
                        verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")

    return test_acc, history

# -------------------------------
# 4. TRAIN ALL MODELS
# -------------------------------
models_dict = {
    "Basic CNN": basic_cnn(),
    "Pooling CNN": pooling_cnn(),
    "Efficient CNN": efficient_cnn(),
    "Dilated CNN": dilated_cnn(),
    "Random Feature CNN": random_feature_cnn()
}

results = {}

for name, model in models_dict.items():
    acc, history = train_model(model, name)
    results[name] = acc

# -------------------------------
# 5. COMPARE RESULTS
# -------------------------------
print("\nFinal Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Plot comparison
plt.figure()
plt.bar(results.keys(), results.values())
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("CNN Architecture Comparison")
plt.show()