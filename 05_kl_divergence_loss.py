# PATTERN: KL Divergence loss for comparing probability distributions
# Learned: why MSE isn't ideal for comparing probability distributions (values
# that must be positive and sum to 1), the KL Divergence formula
# (KL(P||Q) = sum of P(x) * log(P(x)/Q(x))), why PyTorch's nn.KLDivLoss
# expects log-probabilities as its first argument (commonly paired with
# torch.log_softmax in real networks), and softmax as the function that
# converts raw network outputs into valid probability distributions
# no need for torch.log_softmax here

# KEY EXPERIMENT:
# - Similar distributions   -> small KL divergence (0.0731)
# - Very different distributions -> large KL divergence (1.6094)
# This confirms KL divergence correctly measures how different two
# probability distributions are — exactly what's needed to compare a
# neural network's corrected (noisy) quantum output against the true
# (ideal) output.

import torch
import torch.nn as nn

# Example 1: similar distributions (small noise)
ideal = torch.tensor([[0.5, 0.5, 0.0, 0.0]])       # true/ideal circuit output
noisy = torch.tensor([[0.45, 0.48, 0.04, 0.03]])   # noisy circuit output, close to ideal

kl_loss = nn.KLDivLoss(reduction='batchmean')
loss = kl_loss(noisy.log(), ideal)   # note: .log() required on the predicted distribution, for the actual one API handle all that stuff.
                                     # note: .log() is required on the predicted distribution only.
                                     # The API internally handles log(ideal) and all remaining math (subtracting,
                                     # summing, averaging) — this split comes from the KL divergence formula
                                     # algebraically separating into two log terms, one of which only needs Q.
                                     # only one log is needed due to more efficent calculations 
print(f"KL Divergence (similar distributions): {loss.item():.4f}")

# Example 2: very different distributions (heavy noise / bad prediction)
noisy_bad = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
loss_bad = kl_loss(noisy_bad.log(), ideal)
print(f"KL Divergence (very different distributions): {loss_bad.item():.4f}")

# Note: in a real project, a network's raw output would need softmax first:
# raw_output = model(x)
# predicted_log_probs = torch.log_softmax(raw_output, dim=1)
# loss = kl_loss(predicted_log_probs, ideal_target)
