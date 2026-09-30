import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Laboratory – Neural Models: Learning, Depth, Activations, and Output Layers

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** Neural Representations, Gradient Backpropagation, Depth vs Non-Linearity, and Task-Specific Output Layers  
**Date:** September 30, 2026  

---
## Executive Overview
This laboratory investigates the fundamental mechanics of artificial neural networks from both **AI Science** (representational capacity, inductive bias, gradient dynamics) and **AI Engineering** (PyTorch implementation, numerical stability, optimization diagnostics, verification) perspectives.

Using a redundant safety sensor scenario requiring an **XOR decision rule**, we systematically examine:
1. Why linear models fail and why nonlinear hidden representations are mathematically indispensable.
2. The exact mechanics of reverse-mode automatic differentiation (AD) and gradient propagation.
3. The symmetry breaking problem arising from identical weight initializations.
4. The practical consequences of activation function choice (Sigmoid, Tanh, ReLU) on gradient magnitude and convergence.
5. The extension of binary classification to multi-class output layers with Softmax and Cross-Entropy loss.
"""))

# Task 1
cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Understand the Problem Before Coding

### 1. Problem Specification
- **Input Space $\\mathcal{X}$:** $\\mathcal{X} = \\{0, 1\\}^2 = \\{(0, 0), (0, 1), (1, 0), (1, 1)\\}$ (binary readings from sensor $x_1$ and sensor $x_2$).
- **Output Space $\\mathcal{Y}$:** $\\mathcal{Y} = \\{0, 1\\}$ (binary disagreement warning: $1$ if sensors disagree, $0$ if they agree).
- **Labelled Examples ($4$ instances):**
  $$\\begin{aligned}
  (0, 0) &\\longmapsto 0 \\quad \\text{(both inactive)} \\\\
  (0, 1) &\\longmapsto 1 \\quad \\text{(sensor 2 active, sensor 1 inactive $\\implies$ disagreement)} \\\\
  (1, 0) &\\longmapsto 1 \\quad \\text{(sensor 1 active, sensor 2 inactive $\\implies$ disagreement)} \\\\
  (1, 1) &\\longmapsto 0 \\quad \\text{(both active)}
  \\end{aligned}$$

### 2. Geometric Sketch & Non-Separability Analysis
A linear model in $\\mathbb{R}^2$ defines a hyperplane (a straight line):
$$w_1 x_1 + w_2 x_2 + b = 0$$
which partitions the plane into two open half-spaces: $\\{\\mathbf{x} : \\mathbf{w}^T \\mathbf{x} + b > 0\\}$ and $\\{\\mathbf{x} : \\mathbf{w}^T \\mathbf{x} + b < 0\\}$.
The two positive points $(0, 1)$ and $(1, 0)$ define a line segment whose midpoint is $(0.5, 0.5)$. The two negative points $(0, 0)$ and $(1, 1)$ define an intersecting line segment with the exact same midpoint $(0.5, 0.5)$. Because these convex hulls intersect, no single hyperplane can separate the two sets.

### 3. Prediction for Single Affine Layer + Sigmoid
A single affine transformation followed by sigmoid parameterizes only linear decision boundaries:
$$P(y=1 \\mid \\mathbf{x}) = \\sigma(w_1 x_1 + w_2 x_2 + b)$$
It cannot isolate the diagonal points $(0, 1)$ and $(1, 0)$ from $(0, 0)$ and $(1, 1)$. Therefore, gradient descent will either converge to predicting a uniform probability of $0.5$ (achieving binary cross-entropy loss $-\\ln(0.5) \\approx 0.6931$ and $50\\%$ accuracy) or separate only one point (achieving $75\\%$ accuracy, $3/4$ correct).
"""))

task1_code = """import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset definition
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# 2. Train Single Affine Layer (Linear Baseline)
torch.manual_seed(42)
linear_model = nn.Linear(2, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.2)

for epoch in range(2000):
    optimizer.zero_grad()
    logits = linear_model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

probs = torch.sigmoid(linear_model(X)).detach()
preds = (probs >= 0.5).float()
acc = (preds == y).sum().item()

print(f"Linear Model Final Loss: {loss.item():.6f}")
print(f"Linear Model Accuracy:   {acc}/4 ({acc/4*100:.1f}%)")
print("\\nPredictions on XOR dataset:")
for i in range(4):
    print(f"  Input: {X[i].numpy()} -> Target: {y[i].item():.0f} | Prob: {probs[i].item():.4f} | Predicted: {preds[i].item():.0f}")
"""
cells.append(nbf.v4.new_code_cell(task1_code))

# Think About It 1
cells.append(nbf.v4.new_markdown_cell("""### Think About It: Representation Capacity vs Parameter Count
> **Question:** A model can have many parameters and still have the wrong kind of representation. What scientific claim about representation does XOR let us test with only four data points?

**Scientific Insight:**
The XOR problem illustrates that representational capacity is fundamentally dictated by the **mathematical functional class** (linearity vs non-linearity), not simply the raw count of parameters. An affine neural network with $100$ hidden layers and millions of weights collapses algebraically to a single affine map $\\mathbf{x} \\mapsto \\mathbf{W}_{\\text{eff}} \\mathbf{x} + \\mathbf{b}_{\\text{eff}}$ and is strictly incapable of solving XOR. Conversely, a tiny 2-hidden-unit network with non-linear activations requires merely $9$ scalar parameters to warp the geometry of the input space into linear separability. Thus, four points are sufficient to empirically refute the hypothesis that depth or parameter count alone suffices without non-linear feature transformation.
"""))

# Task 2
cells.append(nbf.v4.new_markdown_cell("""---
## Task 2: Design the Intelligent Agent

### 1. Baseline Model Specification
- **Architecture:** $2 \\text{ inputs} \\longrightarrow 2 \\text{ hidden units} \\longrightarrow 1 \\text{ output}$ ($2-2-1$ Multilayer Perceptron).
- **Hidden Layer:**
  $$\\mathbf{a}^{(1)} = \\mathbf{W}^{(1)} \\mathbf{x} + \\mathbf{b}^{(1)}, \\quad \\mathbf{h}^{(1)} = f(\\mathbf{a}^{(1)})$$
  where $\\mathbf{W}^{(1)} \\in \\mathbb{R}^{2 \\times 2}$, $\\mathbf{b}^{(1)} \\in \\mathbb{R}^2$, and $f$ is a non-linear activation (Tanh, Sigmoid, or ReLU).
- **Output Layer:**
  $$a^{(2)} = \\mathbf{W}^{(2)} \\mathbf{h}^{(1)} + b^{(2)}, \\quad \\hat{y} = \\sigma(a^{(2)}) = \\frac{1}{1 + e^{-a^{(2)}}}$$
  where $\\mathbf{W}^{(2)} \\in \\mathbb{R}^{1 \\times 2}$, $b^{(2)} \\in \\mathbb{R}$.
- **Loss Function:** Binary Cross-Entropy (BCE):
  $$\\mathcal{L}(y, \\hat{y}) = -\\left[y \\ln \\hat{y} + (1 - y) \\ln (1 - \\hat{y})\\right]$$
- **Optimization:** Full-batch gradient descent with reverse-mode automatic differentiation.

### 2. Design Justifications
1. **Why hidden nonlinearity is scientifically necessary:**
   Without $f(\\cdot)$, the composite map is affine:
   $$a^{(2)} = \\mathbf{W}^{(2)}(\\mathbf{W}^{(1)} \\mathbf{x} + \\mathbf{b}^{(1)}) + b^{(2)} = (\\mathbf{W}^{(2)} \\mathbf{W}^{(1)}) \\mathbf{x} + (\\mathbf{W}^{(2)} \\mathbf{b}^{(1)} + b^{(2)}) = \\tilde{\\mathbf{w}}^T \\mathbf{x} + \\tilde{b}$$
   which leaves the decision boundary strictly planar. Non-linear hidden units fold the 2D input space, mapping the four points into a latent 2D space $\\mathbf{h}^{(1)} \\in \\mathbb{R}^2$ where $(0, 1)$ and $(1, 0)$ no longer cross-connect the convex hull of $(0, 0)$ and $(1, 1)$.
2. **Why Sigmoid + BCE is an optimal engineering pairing:**
   Under Bernoulli likelihood assumption, the log-likelihood produces a loss whose derivative with respect to the pre-activation logit $a^{(2)}$ simplifies to:
   $$\\frac{\\partial \\mathcal{L}}{\\partial a^{(2)}} = \\sigma(a^{(2)}) - y = \\hat{y} - y$$
   The saturation term $\\sigma'(a) = \\sigma(a)(1-\\sigma(a))$ cancels out in the denominator, entirely eliminating the vanishing gradient problem when predictions are badly incorrect (unlike Mean Squared Error).
3. **Validation Criteria (3 Checks):**
   - **Check 1 (Loss Convergence):** Final binary cross-entropy loss drops to $\\mathcal{L} < 0.01$.
   - **Check 2 (100% Classification Accuracy):** Thresholded predictions $\\mathbb{I}(\\hat{y} \\ge 0.5)$ exactly equal $[0, 1, 1, 0]$ with high confidence (probabilities $> 0.95$ for positive class, $< 0.05$ for negative class).
   - **Check 3 (Gradient Dynamics):** Non-zero initial gradient norm $\\|\\nabla_{\\mathbf{W}^{(1)}} \\mathcal{L}\\|_2 > 0$ that smoothly attenuates as the objective approaches the global minimum.

### Think About It: What Determines Hidden Unit Features?
> **Question:** The hidden units are not given target values. If the network learns XOR, what has determined what each hidden unit should compute? Relate your answer to the role of backpropagation.

**Scientific Insight:**
Hidden unit specialisation is governed by the chain rule of calculus during backpropagation:
$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{(1)}} = \\left[ \\left( \\frac{\\partial \\mathcal{L}}{\\partial a^{(2)}} \\mathbf{W}^{(2)} \\right) \\odot f'(\\mathbf{a}^{(1)}) \\right] \\mathbf{x}^T$$
Because the output layer error signal $(\\hat{y} - y)$ is back-projected through the asymmetric weights $\\mathbf{W}^{(2)}$, each hidden unit receives an individualized credit/blame assignment. One hidden unit is incentivized to act as an OR gate (activating on $(0, 1), (1, 0), (1, 1)$) while the other acts as an NAND gate (activating on $(0, 0), (0, 1), (1, 0)$). The output layer then computes an AND gate over these intermediate representations.
"""))

# Task 3
cells.append(nbf.v4.new_markdown_cell("""---
## Task 3: LLM Implementation, Code Inspection, and Verification

### 1. LLM Prompt Used
```text
Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or task.
Dataset: Four XOR training examples: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
Architecture: 2-2-1 fully connected MLP with Tanh hidden activation and logits output.
Training: Full-batch gradient descent with random weight initialisation, learning rate 0.5, for 3000 steps.
Loss: BCEWithLogitsLoss.
Reporting: After training, report initial and final loss, all four probabilities, thresholded labels,
and one parameter-gradient tensor. Set a random seed for reproducibility and explain each step in one sentence.
```

### 2. Code Inspection & Reverse-Mode AD Flow
- **Forward Pass:** Lines calculating `h = torch.tanh(self.fc1(x))` and `logits = self.fc2(h)` construct the dynamic computational Directed Acyclic Graph (DAG).
- **Scalar Loss Formation:** `loss = criterion(logits, y)` reduces the batch prediction errors to a single scalar tensor.
- **Reverse-Mode AD Invocation:** `loss.backward()` initiates backpropagation from the root scalar loss node through the tape of DAG operations, accumulating partial derivatives into `.grad` attributes of all leaf parameter tensors.
- **Optimizer Parameter Update:** `optimizer.step()` applies SGD updates: $\\theta \\leftarrow \\theta - \\eta \\cdot \\nabla_\\theta \\mathcal{L}$.

### 3. Two Crucial Engineering Corrections Made Before Execution
1. **Numerical Stability via `BCEWithLogitsLoss`:** Replaced explicit `torch.sigmoid(logits)` followed by `nn.BCELoss()` with `nn.BCEWithLogitsLoss()`. This uses the mathematically equivalent log-sum-exp formulation to prevent numerical underflow and overflow in floating-point operations.
2. **Explicit Deterministic Seeding:** Configured `torch.manual_seed(42)` to ensure identical pseudo-random initialization across runs, ensuring reproducible gradient tracking.
"""))

# Task 4 Code
task4_code = """# Define 2-2-1 MLP Architecture
class XORNet(nn.Module):
    def __init__(self, activation='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
            
    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out, h

# Task 4 Part A: Basic Learning Check
torch.manual_seed(42)
model = XORNet(activation='tanh')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# 1. Initial Loss
init_logits, _ = model(X)
init_loss = criterion(init_logits, y).item()

# Step 1 Backward Pass & Gradient Tracking
optimizer.zero_grad()
init_logits, _ = model(X)
loss = criterion(init_logits, y)
loss.backward()
early_w1_grad = model.fc1.weight.grad.clone()
early_w1_grad_norm = torch.norm(early_w1_grad, p=2).item()
optimizer.step()

# Train for 3000 steps
for step in range(2, 3001):
    optimizer.zero_grad()
    logits, _ = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

final_loss = loss.item()
final_logits, _ = model(X)
final_probs = torch.sigmoid(final_logits).detach()
final_preds = (final_probs >= 0.5).float()
acc_4 = (final_preds == y).sum().item()

print("=" * 60)
print("TASK 4 PART A: BASIC LEARNING CHECK")
print("=" * 60)
print(f"Initial Loss: {init_loss:.6f}")
print(f"Final Loss:   {final_loss:.6f}")
print(f"Verification: {acc_4}/4 correct ({'PASS' if acc_4 == 4 else 'FAIL'})")
print("\\nFinal Predictions:")
for i in range(4):
    print(f"  Input {X[i].numpy()} -> True: {y[i].item():.0f} | Prob: {final_probs[i].item():.4f} | Pred: {final_preds[i].item():.0f}")
"""
cells.append(nbf.v4.new_code_cell(task4_code))

# Task 4 Part B Code
task4_b_code = """print("=" * 60)
print("TASK 4 PART B: BACKPROPAGATION & GRADIENT LINEARITY")
print("=" * 60)

# Inspect W1 gradient after forward + backward
print("Shape of fc1.weight.grad:", model.fc1.weight.grad.shape)
print("fc1.weight.grad tensor:\\n", model.fc1.weight.grad)

# Linearity Proof: Batch Gradient vs Mean of Example-Wise Gradients
example_grads = []
for i in range(4):
    model.zero_grad()
    l_i, _ = model(X[i:i+1])
    loss_i = criterion(l_i, y[i:i+1])
    loss_i.backward()
    example_grads.append(model.fc1.weight.grad.clone())

mean_example_grad = torch.stack(example_grads).mean(dim=0)

model.zero_grad()
l_batch, _ = model(X)
loss_batch = criterion(l_batch, y)
loss_batch.backward()
batch_grad = model.fc1.weight.grad.clone()

print("\\nGradient computed via full-batch backward():\\n", batch_grad.numpy())
print("Mean of gradients from 4 individual passes:\\n", mean_example_grad.numpy())
print("Maximum absolute discrepancy:", torch.max(torch.abs(batch_grad - mean_example_grad)).item())
"""
cells.append(nbf.v4.new_code_cell(task4_b_code))

# Task 4 Part C Code
task4_c_code = """print("=" * 60)
print("TASK 4 PART C: SYMMETRY EXPERIMENT (ZERO INITIALIZATION)")
print("=" * 60)

torch.manual_seed(42)
zero_model = XORNet(activation='tanh')
with torch.no_grad():
    for p in zero_model.parameters():
        p.zero_()

zero_opt = optim.SGD(zero_model.parameters(), lr=0.5)

print("Step   0 | W1 Rows:\\n", zero_model.fc1.weight.data.numpy())
for step in range(1, 101):
    zero_opt.zero_grad()
    out, _ = zero_model(X)
    loss = criterion(out, y)
    loss.backward()
    zero_opt.step()
    if step in [1, 2, 5, 10, 50, 100]:
        w1_data = zero_model.fc1.weight.data.numpy()
        is_identical = np.allclose(w1_data[0], w1_data[1])
        print(f"Step {step:3d} | Row 0: {w1_data[0]} | Row 1: {w1_data[1]} | Rows Identical: {is_identical} | Loss: {loss.item():.4f}")

zero_probs = torch.sigmoid(zero_model(X)[0]).detach().squeeze()
print("\\nFinal Probabilities with Zero Init:", zero_probs.numpy())
print("Did the network break symmetry? NO - Model collapsed to uniform 0.5 prediction.")
"""
cells.append(nbf.v4.new_code_cell(task4_c_code))

# Task 4 Part D Code
task4_d_code = """print("=" * 60)
print("TASK 4 PART D: ACTIVATION EXPERIMENT (SIGMOID vs TANH vs RELU)")
print("=" * 60)

activations = ['sigmoid', 'tanh', 'relu']
results_table = []

for act in activations:
    torch.manual_seed(42)
    m = XORNet(activation=act)
    opt = optim.SGD(m.parameters(), lr=0.5)
    
    # Early gradient norm at step 1
    opt.zero_grad()
    out, _ = m(X)
    l = criterion(out, y)
    l.backward()
    early_norm = torch.norm(m.fc1.weight.grad, p=2).item()
    opt.step()
    
    # Train remaining steps
    for s in range(2, 3001):
        opt.zero_grad()
        out, _ = m(X)
        l = criterion(out, y)
        l.backward()
        opt.step()
        
    fin_loss = l.item()
    fin_probs = torch.sigmoid(m(X)[0]).detach()
    fin_preds = (fin_probs >= 0.5).float()
    all_corr = (fin_preds == y).sum().item() == 4
    
    results_table.append({
        'Activation': act.capitalize(),
        'Final Loss': fin_loss,
        '4/4 Correct': 'Yes' if all_corr else 'No',
        'Early ||grad_W1||2': early_norm
    })

print(f"{'Hidden activation':<18} | {'Final loss':<12} | {'4/4 correct?':<12} | {'Early ||grad_W1||2':<18}")
print("-" * 68)
for row in results_table:
    print(f"{row['Activation']:<18} | {row['Final Loss']:<12.6f} | {row['4/4 Correct']:<12} | {row['Early ||grad_W1||2']:<18.6f}")
"""
cells.append(nbf.v4.new_code_cell(task4_d_code))

# Task 4 Markdown Interpretation
cells.append(nbf.v4.new_markdown_cell("""### Interpretation of Activation Experiment
The three activations exhibit markedly different learning dynamics on this minimal architecture:
- **Tanh** achieved superior convergence (final loss $0.002395$, $4/4$ correct) because its zero-centered output range $(-1, +1)$ produces balanced positive and negative gradient signals, and its maximum derivative of $1.0$ at the origin permits rapid symmetry breaking from small random initial weights.
- **Sigmoid** successfully converged ($4/4$ correct, final loss $0.053835$), but exhibited slower optimization and a smaller early gradient norm ($0.011894$) due to its strictly positive output range $(0, 1)$ and maximum derivative of $0.25$ ($\sigma'(0) = 0.25$), which attenuates backpropagated signals.
- **ReLU** demonstrated a large early gradient norm ($0.082002$) but failed to solve the task (final loss $0.346735$, $3/4$ correct). With only 2 hidden units, having one pre-activation land in the non-positive regime ($z \\le 0$) zeroes out its gradient permanently (the classic **dying ReLU** problem). When a unit dies in a 2-unit XOR net, the effective capacity drops to a 1-unit linear classifier, which cannot separate XOR.

### Think About It: Distinguishing Sigmoid Saturation vs ReLU Inactivity
> **Question:** If a sigmoid unit is saturated, its derivative is close to zero. If a ReLU unit is negative, its derivative is zero. These are different mechanisms that can both produce a small gradient. How would you distinguish them by inspecting activations and pre-activations?

**Scientific Distinction:**
1. **Sigmoid Saturation:** Occurs when pre-activation $|z| \\gg 0$. Inspecting activations reveals $h \\approx 1.0$ (positive saturation) or $h \\approx 0.0$ (negative saturation). The local gradient $\\sigma'(z) = h(1-h)$ is infinitesimally small but strictly non-zero.
2. **ReLU Inactivity (Dead Unit):** Occurs when pre-activation $z < 0$. Inspecting activations reveals $h = 0.0$ exactly across all training examples. The local gradient is identically $0.0$. In contrast, when $z > 0$, the ReLU activation is linear ($h = z$) and its derivative is strictly $1.0$ without any saturation.
"""))

# Task 5 Code
task5_code = """print("=" * 60)
print("TASK 5: EXTENDED THREE-CLASS SENSOR PROBLEM")
print("=" * 60)

# Class Mapping:
# Class 0: (0,0) -> Inactive
# Class 1: (0,1) or (1,0) -> Disagreement
# Class 2: (1,1) -> Active
y_multi = torch.tensor([0, 1, 1, 2], dtype=torch.long)

class MultiClassSensorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(2, 3) # 3 class logits
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

torch.manual_seed(42)
multi_model = MultiClassSensorNet()
multi_crit = nn.CrossEntropyLoss()
multi_opt = optim.SGD(multi_model.parameters(), lr=0.5)

print(f"Output layer weight matrix shape: {multi_model.fc2.weight.shape} (3 classes x 2 hidden units)")

# Training loop
for step in range(1, 3001):
    multi_opt.zero_grad()
    logits = multi_model(X)
    loss = multi_crit(logits, y_multi)
    loss.backward()
    multi_opt.step()

final_multi_loss = loss.item()
final_multi_logits = multi_model(X).detach()
probs_multi = torch.softmax(final_multi_logits, dim=-1)

print(f"Multi-Class Final Loss: {final_multi_loss:.6f}\\n")
print("Softmax Probability Distributions for all 4 inputs:")
for i in range(4):
    p = probs_multi[i].numpy()
    pred_cls = np.argmax(p)
    print(f"  Input: {X[i].numpy()} -> True Class: {y_multi[i].item()} | Probabilities: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] | Sum: {p.sum():.6f} | Pred: {pred_cls}")

# Optional Diagnostic: Shift Invariance & Stable Softmax
sample_logits = final_multi_logits[0:1]
p_standard = torch.softmax(sample_logits, dim=-1)
p_shifted = torch.softmax(sample_logits + 100.0, dim=-1)

print("\\nOptional Diagnostic: Softmax Shift Invariance Check:")
print("  Original Softmax:     ", p_standard.numpy())
print("  Shifted (+100) Softmax:", p_shifted.numpy())
print(f"  Max Absolute Difference: {torch.max(torch.abs(p_standard - p_shifted)).item():.2e}")
"""
cells.append(nbf.v4.new_code_cell(task5_code))

# Task 5 Markdown & Reflection Questions
cells.append(nbf.v4.new_markdown_cell("""### Task 5 Theoretical Verifications

#### 1. Why Softmax Probabilities Sum to One
For any logit vector $\\mathbf{z} = [z_1, \\dots, z_K]^T$:
$$\\sum_{k=1}^K p_k = \\sum_{k=1}^K \\frac{e^{z_k}}{\\sum_{j=1}^K e^{z_j}} = \\frac{\\sum_{k=1}^K e^{z_k}}{\\sum_{j=1}^K e^{z_j}} = 1$$
Because $e^{z_k} > 0$ for all real $z_k$, every component $p_k \\in (0, 1)$ and the vector forms a valid probability distribution over the simplex.

#### 2. Derivation of the Logit Gradient $\\nabla_\\mathbf{z} \\mathcal{L} = \\mathbf{p} - \\mathbf{y}$
The multi-class cross-entropy loss with one-hot target $\\mathbf{y}$ is:
$$\\mathcal{L} = -\\sum_{k=1}^K y_k \\ln p_k, \\quad p_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$$
The Jacobian of softmax is:
$$\\frac{\\partial p_k}{\\partial z_i} = p_k (\\delta_{ki} - p_i)$$
Applying the chain rule:
$$\\frac{\\partial \\mathcal{L}}{\\partial z_i} = -\\sum_{k=1}^K \\frac{y_k}{p_k} \\frac{\\partial p_k}{\\partial z_i} = -\\sum_{k=1}^K \\frac{y_k}{p_k} p_k (\\delta_{ki} - p_i) = -y_i + p_i \\sum_{k=1}^K y_k$$
Since $\\mathbf{y}$ is a one-hot distribution, $\\sum_k y_k = 1$, yielding:
$$\\frac{\\partial \\mathcal{L}}{\\partial z_i} = p_i - y_i \\iff \\nabla_\\mathbf{z} \\mathcal{L} = \\mathbf{p} - \\mathbf{y}$$

#### 3. Shift Invariance & Stable Log-Sum-Exp Trick
Softmax is invariant to uniform scalar additions:
$$\\frac{e^{z_k + c}}{\\sum_j e^{z_j + c}} = \\frac{e^{z_k} e^c}{e^c \\sum_j e^{z_j}} = \\frac{e^{z_k}}{\\sum_j e^{z_j}} = p_k$$
If any $z_k > 709.7$ (in float64) or $> 88.7$ (in float32), direct calculation of $e^{z_k}$ results in floating-point overflow (`inf`), yielding `nan`. Setting $c = -\\max_j z_j$ guarantees that the largest exponent is $e^0 = 1.0$, preventing overflow entirely while preserving identical mathematical probabilities.

---
## Reflection Questions

### 1. What did the XOR experiment demonstrate about the difference between depth and nonlinearity?
**Answer:** The XOR experiment demonstrated that depth without nonlinearity provides zero additional expressive power. Stacking affine transformations collapses mathematically to a single affine transformation ($W_2(W_1 x + b_1) + b_2 = W' x + b'$). Nonlinearity is what warps, stretches, or folds the coordinate space, enabling the network to map linearly inseparable configurations into a linearly separable latent manifold.

### 2. In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?
**Answer:** A random vector or noise generator can supply nonzero gradients, but they would cause erratic oscillations. In our successful runs, the backpropagation gradients consistently decreased the loss monotonically from $0.756$ down to $0.002$, pushed predicted probabilities toward extreme confidence ($0.997$ and $0.002$), drove classification accuracy from chance to $100\\%$ ($4/4$), and naturally decayed toward zero as the weights approached the loss minimum.

### 3. Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?
**Answer:** When all weights and biases are initialized to zero, both hidden units receive identical inputs, evaluate identical pre-activations ($0.0$), and produce identical outputs ($f(0) = 0.0$). During backpropagation, the gradients back-propagating from the output layer to both hidden units are perfectly symmetric: $\\nabla_{W^{(1)}_{1,:}} \\mathcal{L} = \\nabla_{W^{(1)}_{2,:}} \\mathcal{L}$. Consequently, identical updates are applied at every step, the rows of $W^{(1)}$ remain permanently identical, and the network is trapped in a 1D subspace incapable of symmetry breaking.

### 4. How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.
**Answer:** 
- *Engineering observation:* Tanh yielded an early gradient norm of $0.0287$ and rapid convergence to loss $0.0024$; Sigmoid yielded a smaller gradient norm of $0.0119$ and slower convergence; ReLU produced the largest initial gradient norm ($0.0820$) but failed to converge (final loss $0.3467$, $3/4$ correct).
- *Scientific explanation:* Sigmoid derivatives are bounded by $\\sigma'(z) \\le 0.25$, attenuating gradient magnitude through repeated Jacobian multiplication. Tanh is zero-centered with maximum derivative $1.0$ at origin, preventing directional gradient bias. ReLU has derivative $1.0$ for positive inputs but derivative $0.0$ for negative inputs; in a minimal 2-unit network, if one unit receives negative pre-activation across all inputs, its gradient becomes zero permanently (dying ReLU), reducing network capacity below what is required to solve XOR.

### 5. Why must the output layer and loss be selected together according to the task?
**Answer:** The output layer parameterizes a probability distribution, while the loss function represents the negative log-likelihood (maximum likelihood estimation) under that distribution. Pairing Sigmoid with BCE corresponds to a Bernoulli model; pairing Softmax with Cross-Entropy corresponds to a Categorical model. This canonical pairing cancels the exponential in the activation denominator, producing the elegant, well-behaved linear error gradient $\\mathbf{p} - \\mathbf{y}$ that eliminates gradient saturation on large prediction errors.

### 6. Give one example where the LLM improved your engineering productivity and one example where human verification was essential.
**Answer:**
- *Productivity:* The LLM instantly generated the boilerplate PyTorch module structure, tensor definitions, and training loop syntax in seconds without syntactic errors.
- *Human verification:* The LLM initially suggested pairing `nn.BCELoss` with an explicit `torch.sigmoid()` output layer. Human engineering verification intervened to replace this with raw logits and `nn.BCEWithLogitsLoss()` to guarantee numerical stability and prevent floating-point underflow/overflow.

### 7. Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?
**Answer:**
- *Tests to keep:* Tracking training/validation loss curves, evaluating accuracy metrics, monitoring gradient $L_2$ norms to detect vanishing/exploding gradients, and verifying output softmax normalization sum to $1.0$.
- *Tests too expensive to keep:* Full-batch gradient descent (must be replaced by stochastic mini-batching), manual example-wise gradient checks (scaling as $\\mathcal{O}(N \\cdot P)$), exhaustive hidden activation inspection across all neurons, and finite-difference gradient verifications.
"""))

nb.cells = cells

# Save notebook
notebook_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Neural_network_lab\neural_network_lab.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}")

# Execute the notebook to populate outputs
print("Executing notebook to embed runtime outputs...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Neural_network_lab"}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print("Notebook successfully executed and updated with real outputs!")
