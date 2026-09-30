import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: XOR Geometric Inseparability
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.scatter([0, 1], [0, 1], color='#e74c3c', s=160, edgecolors='black', linewidth=1.5, zorder=5, label='Class 0 (y=0: Agree)')
ax.scatter([0, 1], [1, 0], color='#2ecc71', s=160, edgecolors='black', linewidth=1.5, marker='s', zorder=5, label='Class 1 (y=1: Disagree)')

# Connect diagonals
ax.plot([0, 1], [1, 0], 'g--', alpha=0.5, label='Class 1 Diagonal')
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Class 0 Diagonal')

# Illustrative linear boundary attempt
x_vals = np.linspace(-0.2, 1.2, 100)
ax.plot(x_vals, -x_vals + 1.0, 'b-', linewidth=2, label='Failed Linear Boundary')

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xlabel('$x_1$ (Sensor 1)', fontsize=12, fontweight='bold')
ax.set_ylabel('$x_2$ (Sensor 2)', fontsize=12, fontweight='bold')
ax.set_title('Task 1: XOR Decision Problem in $(x_1, x_2)$ Plane', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('xor_problem_plot.png')
plt.close()
print("Saved xor_problem_plot.png")

# Figure 2: Activation Training Curves
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
        return self.fc2(self.act(self.fc1(x)))

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
criterion = nn.BCEWithLogitsLoss()

loss_histories = {}
for act_name, color in [('sigmoid', '#3498db'), ('tanh', '#e67e22'), ('relu', '#9b59b6')]:
    torch.manual_seed(42)
    model = XORNet(activation=act_name)
    opt = optim.SGD(model.parameters(), lr=0.5)
    losses = []
    for step in range(3000):
        opt.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    loss_histories[act_name] = losses

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
for act_name, color in [('sigmoid', '#3498db'), ('tanh', '#e67e22'), ('relu', '#9b59b6')]:
    ax.plot(loss_histories[act_name], label=f'{act_name.capitalize()} (Final: {loss_histories[act_name][-1]:.4f})', color=color, linewidth=2)

ax.set_xlabel('Training Step (Iteration)', fontsize=11, fontweight='bold')
ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax.set_title('Task 4 Part D: Loss Convergence Across Hidden Activations', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('activation_loss_curves.png')
plt.close()
print("Saved activation_loss_curves.png")
