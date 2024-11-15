import random

import numba
import time
import minitorch

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x):
        # Use matrix multiplication directly instead of view/sum operations
        return (x @ self.weights.value) + self.bias.value


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        # self.model = Network(self.hidden_layers, self.backend)
        # optim = minitorch.SGD(self.model.parameters(), learning_rate)
        # BATCH = 10
        # losses = []

        # for epoch in range(max_epochs):
        #     total_loss = 0.0
        #     c = list(zip(data.X, data.y))
        #     random.shuffle(c)
        #     X_shuf, y_shuf = zip(*c)

        #     for i in range(0, len(X_shuf), BATCH):
        #         optim.zero_grad()
        #         X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
        #         y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)
        #         # Forward

        #         out = self.model.forward(X).view(y.shape[0])
        #         prob = (out * y) + (out - 1.0) * (y - 1.0)
        #         loss = -prob.log()
        #         (loss / y.shape[0]).sum().view(1).backward()

        #         total_loss = loss.sum().view(1)[0]

        #         # Update
        #         optim.step()

        #     losses.append(total_loss)
        #     # Logging
        #     if epoch % 10 == 0 or epoch == max_epochs:
        #         X = minitorch.tensor(data.X, backend=self.backend)
        #         y = minitorch.tensor(data.y, backend=self.backend)
        #         out = self.model.forward(X).view(y.shape[0])
        #         y2 = minitorch.tensor(data.y)
        #         correct = int(((out.detach() > 0.5) == y2).sum()[0])
        #         log_fn(epoch, total_loss, correct, losses)
         
        # total_time = time.time() - total_start  
        # print(f"\nTraining completed in {total_time:.2f}s")
        # print(f"Average epoch time: {total_time/max_epochs:.3f}s")
        # print(f"Best accuracy achieved: {best_accuracy:.2f}%")

        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        # Dynamically set batch size for better utilization
        BATCH = 64 if self.backend.__class__.__name__ == "CudaOps" else 32
        
        # Pre-load data to GPU memory to avoid repeated data transfer overhead
        # Shuffle data and convert entire dataset to tensors
        c = list(zip(data.X, data.y))
        random.shuffle(c)
        X_shuf, y_shuf = zip(*c)
        X_all = minitorch.tensor(X_shuf, backend=self.backend)  # Entire dataset on GPU
        y_all = minitorch.tensor(y_shuf, backend=self.backend)  # Entire dataset on GPU

        # Set up tracking for losses
        losses = []

        print(f"Training on {self.backend.__class__.__name__}")
        total_start = time.time()

        # Training loop
        for epoch in range(max_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            
            # Process data in batches
            for i in range(0, len(X_shuf), BATCH):
                batch_end = min(i + BATCH, len(X_shuf))
                optim.zero_grad()  # Zero gradients for each batch

                # Slice the pre-loaded GPU tensors for the current batch
                X_batch = X_all[i:batch_end]
                y_batch = y_all[i:batch_end]

                # Forward pass
                out = self.model.forward(X_batch).view(y_batch.shape[0])

                # Compute loss
                prob = (out * y_batch) + (out - 1.0) * (y_batch - 1.0)
                loss = -prob.log()

                # Backward pass and parameter update
                (loss / y_batch.shape[0]).sum().view(1).backward()
                optim.step()

                # Accumulate batch loss
                total_loss += loss.sum().view(1)[0]

            losses.append(total_loss)

            # Logging and evaluation every 10 epochs
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                out = self.model.forward(X_all).view(y_all.shape[0])  # Forward on full dataset
                correct = int(((out.detach() > 0.5) == y_all).sum()[0])  # Accuracy calculation
                log_fn(epoch, total_loss, correct, losses)
                print(f"Epoch {epoch} completed in {time.time() - epoch_start:.3f}s")

        total_time = time.time() - total_start
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Average epoch time: {total_time / max_epochs:.3f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    FastTrain(
        HIDDEN, backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    ).train(data, RATE)