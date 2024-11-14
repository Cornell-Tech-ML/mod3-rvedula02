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
        
        # Improved initialization scale
        self.layer1 = Linear(2, hidden, backend, scale=0.1)
        self.layer2 = Linear(hidden, hidden, backend, scale=0.1)
        self.layer3 = Linear(hidden, 1, backend, scale=0.1)

    def forward(self, x):
        # Use ReLU for intermediate layers
        h = self.layer1(x).relu()
        h = self.layer2(h).relu()
        return self.layer3(h).sigmoid()  # Keep sigmoid for final layer

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend, scale=1.0):
        super().__init__()
        # Xavier initialization
        bound = scale * (2.0 / (in_size + out_size)) ** 0.5
        self.weights = RParam(in_size, out_size, backend=backend) * bound
        self.bias = minitorch.Parameter(minitorch.zeros((out_size,), backend=backend))
        self.out_size = out_size

    def forward(self, x):
        return x @ self.weights.value + self.bias.value

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
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 64  # Increased batch size
        losses = []
        best_accuracy = 0
        patience = 20
        patience_counter = 0
        
        print(f"Training on {self.backend.__class__.__name__}")
        total_start = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            
            # Learning rate decay
            current_lr = learning_rate * (0.95 ** (epoch // 50))
            for param_group in optim.parameters():
                param_group.learning_rate = current_lr
            
            # Shuffle data
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)
            
            # Batch processing
            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                
                # Get batch
                batch_X = X_shuf[i : min(i + BATCH, len(X_shuf))]
                batch_y = y_shuf[i : min(i + BATCH, len(X_shuf))]
                
                # Convert to tensors
                X = minitorch.tensor(batch_X, backend=self.backend)
                y = minitorch.tensor(batch_y, backend=self.backend)
                
                # Forward pass
                out = self.model.forward(X).view(y.shape[0])
                
                # Binary cross entropy loss
                eps = 1e-7  # Prevent log(0)
                out = out.clamp(eps, 1 - eps)
                loss = -(y * out.log() + (1 - y) * (1 - out).log()).sum()
                
                # Backward pass
                (loss / y.shape[0]).backward()
                
                # Update total loss
                total_loss += loss.detach()[0] / y.shape[0]
                
                # Update weights with gradient clipping
                for p in self.model.parameters():
                    if p.value.grad is not None:
                        p.value.grad.clamp_(-1, 1)
                optim.step()
            
            losses.append(total_loss)
            epoch_time = time.time() - epoch_start
            
            # Evaluation
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                accuracy = (correct/len(data.y))*100
                
                log_fn(epoch, total_loss, correct, losses)
                print(f"Epoch time: {epoch_time:.3f}s, Accuracy: {accuracy:.2f}%, Learning rate: {current_lr:.6f}")
                
                # Early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. Best accuracy: {best_accuracy:.2f}%")
                    break
        
            total_time = time.time() - total_start
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Average epoch time: {total_time/max_epochs:.3f}s")
            print(f"Best accuracy achieved: {best_accuracy:.2f}%


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
        data = minitorch.datasets["Simple"].simple(PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    FastTrain(
        HIDDEN, backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    ).train(data, RATE)
