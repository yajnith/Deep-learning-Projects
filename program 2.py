import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights (2-2-1 network)
np.random.seed(0)
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

# Activation
def sigmoid(x): return 1/(1+np.exp(-x))
def d_sigmoid(x): return x*(1-x)

# Training (Gradient Descent)
lr = 0.1
for epoch in range(10000):

    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)          # hidden layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)          # output

    # Loss
    loss = np.mean((y - a2)**2)

    # Backprop
    d2 = (y - a2) * d_sigmoid(a2)
    d1 = d2.dot(W2.T) * d_sigmoid(a1)

    # Update
    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final Output
print("\nPredictions:")
print(np.round(a2))