# pytorch-learning-log
Hands-on PyTorch fundamentals — training loops, gradients, hidden layers, and batching — built while preparing for a quantum computing + deep learning internship project.

#PyTorch Learning Log

Hands-on exercises building up core PyTorch fundamentals before starting 
a quantum computing + deep learning internship project (noise mitigation 
using neural networks).

## Contents

1. **`01_manual_gradient_descent.py`** — raw PyTorch, no `nn.Module`, manual 
   gradient descent from scratch. Learning `y = 2x` using tensors, 
   `requires_grad`, and manually applying `.backward()` and gradient updates.

2. **`02_nn_module_basic.py`** — same problem, rebuilt using `nn.Module`, 
   `nn.Linear`, `nn.MSELoss`, and `torch.optim` — the standard, real-world 
   PyTorch pattern.

3. **`03_hidden_layer_relu.py`** — adding a hidden layer (8 neurons) with 
   ReLU activation, turning the model from simple linear regression into 
   an actual small neural network.

4. **`04_batching_dataloader.py`** — full training pipeline using 
   `TensorDataset` + `DataLoader` for batching, plus a hands-on comparison 
   between SGD and Adam optimizers at different learning rates.

## Key lessons learned

- Learning rate matters more than optimizer choice alone — a well-tuned 
  SGD outperformed both tested Adam configurations on this problem.
- Adam isn't automatically better than SGD — every optimizer still needs 
  its learning rate tuned for the specific problem.
- Batching introduces noise into training (loss bounces batch to batch), 
  which is normal and expected, not a bug.
- Without a nonlinear activation function (like ReLU), stacking multiple 
  linear layers is mathematically identical to having just one layer — 
  the network gains no real extra capacity.
- `requires_grad=True` is what allows PyTorch to automatically compute 
  gradients via `.backward()`; without it, `.grad` stays `None`.

## Next steps

- Learn Qiskit for building and simulating quantum circuits
- Apply this same training pattern to real (simulated) noisy quantum data
- Explore KL Divergence loss for comparing probability distributions
