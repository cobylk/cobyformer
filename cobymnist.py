import numpy as np
from sklearn.datasets import fetch_openml

from cobynn import MLP
from cobyoptim import AdamW, CosineWithLinearWarmup
from cobyutils import CrossEntropyLoss

model = MLP(d_in=784, d_hidden=128, d_out=10, n_hidden=2)
criterion = CrossEntropyLoss()
optimizer = AdamW([{"params": list(model.parameters())}])
scheduler = CosineWithLinearWarmup(optimizer, 65_000, lr_max=1e-3, lr_min=1e-4)


mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.reshape(-1, 28, 28) / 255.0
Y = mnist.target.astype(int)

for epoch in range(10):
    epoch_loss = 0
    correct = 0
    for i in range(0, 65_000, 100):
        x = X[i : i + 100]
        y = Y[i : i + 100]
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        correct += (np.argmax(output.data, axis=-1) == y).sum()
        epoch_loss += loss.data
    print(
        f"epoch {epoch:02d} --- loss: {(epoch_loss / 65_000):.4f} | acc: {(correct / 65_000):.4f}"
    )

    correct = 0
    val_loss = 0
    for i in range(65_000, 70_000, 100):
        x = X[i : i + 100]
        y = Y[i : i + 100]
        output = model(x)
        loss = criterion(output, y)
        correct += (np.argmax(output.data, axis=-1) == y).sum()
        val_loss += loss.data
    print(f"validation - loss: {(val_loss / 5_000):.4f} | acc: {(correct / 5_000):.4f}")
