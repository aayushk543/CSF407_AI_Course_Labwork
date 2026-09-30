import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_all():
    print("=" * 70)
    print("TASK 1: LINEAR MODEL BASELINE (Predicting failure on XOR)")
    print("=" * 70)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    torch.manual_seed(42)
    linear_model = nn.Linear(2, 1)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.SGD(linear_model.parameters(), lr=0.1)
    
    for epoch in range(2000):
        opt.zero_grad()
        out = linear_model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        
    probs = torch.sigmoid(linear_model(X)).detach()
    preds = (probs >= 0.5).float()
    print("Linear Model Final Loss:", loss.item())
    print("Linear Model Predictions:")
    for i in range(4):
        print(f"  Input: {X[i].numpy()} -> True: {y[i].item():.0f}, Prob: {probs[i].item():.4f}, Pred: {preds[i].item():.0f}")
    correct = (preds == y).sum().item()
    print(f"Linear Model Accuracy: {correct}/4 ({correct/4*100:.1f}%)")

    print("\n" + "=" * 70)
    print("TASK 3 & 4 (PART A): 2-2-1 MLP EXPERIMENT (Binary XOR)")
    print("=" * 70)
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
            else:
                raise ValueError("Unknown activation")
                
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out, h

    torch.manual_seed(42)
    model = XORNet(activation='tanh')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    # Initial loss
    initial_logits, _ = model(X)
    initial_loss = criterion(initial_logits, y).item()
    print(f"Initial Loss: {initial_loss:.6f}")

    # Record early gradient
    optimizer.zero_grad()
    initial_logits, _ = model(X)
    loss = criterion(initial_logits, y)
    loss.backward()
    early_grad_norm = torch.norm(model.fc1.weight.grad, p=2).item()
    early_grad_w1 = model.fc1.weight.grad.clone()
    print(f"Step 1 Grad Norm ||grad(W1)||2: {early_grad_norm:.6f}")
    print(f"Step 1 Grad tensor W1.grad:\n{early_grad_w1.numpy()}")
    optimizer.step()

    # Continue training for 3000 steps
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

    print(f"Final Loss after 3000 steps: {final_loss:.6f}")
    print("Final Predictions:")
    for i in range(4):
        print(f"  Input: {X[i].numpy()} -> True: {y[i].item():.0f}, Prob: {final_probs[i].item():.4f}, Pred: {final_preds[i].item():.0f}")
    acc = (final_preds == y).sum().item()
    print(f"Verification: {acc}/4 correct: {'PASS' if acc == 4 else 'FAIL'}")

    print("\n" + "=" * 70)
    print("TASK 4 (PART B): BACKPROPAGATION CHECK & GRADIENT ANALYSIS")
    print("=" * 70)
    # Check manual mean gradient vs backward
    print("W1 weight shape:", model.fc1.weight.shape)
    print("W1 weight grad:\n", model.fc1.weight.grad)
    print("Explanation of parameter.grad: represents dL/dW^(1) = (1/N) * sum_i (dL_i / dW^(1)).")
    
    # Calculate example-wise gradients to numerically demonstrate linearity of gradients
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
    
    print("Batch gradient computed by backward():\n", batch_grad.numpy())
    print("Average of 4 individual example gradients:\n", mean_example_grad.numpy())
    print("Max difference:", torch.max(torch.abs(batch_grad - mean_example_grad)).item())

    print("\n" + "=" * 70)
    print("TASK 4 (PART C): SYMMETRY EXPERIMENT (Zero-initialized Weights)")
    print("=" * 70)
    torch.manual_seed(42)
    zero_model = XORNet(activation='tanh')
    with torch.no_grad():
        for p in zero_model.parameters():
            p.zero_()

    zero_opt = optim.SGD(zero_model.parameters(), lr=0.5)
    print("Initial W1 (Step 0):\n", zero_model.fc1.weight.data.numpy())

    for s in range(1, 101):
        zero_opt.zero_grad()
        out, _ = zero_model(X)
        l = criterion(out, y)
        l.backward()
        zero_opt.step()
        if s in [1, 2, 5, 10, 50, 100]:
            w1 = zero_model.fc1.weight.data.numpy()
            print(f"Step {s:3d} | Row 0: {w1[0]}, Row 1: {w1[1]} | Identical: {np.allclose(w1[0], w1[1])} | Loss: {l.item():.4f}")

    final_zero_probs = torch.sigmoid(zero_model(X)[0]).detach()
    print("Zero-init final probabilities:", final_zero_probs.squeeze().numpy())
    print("Did zero-init break symmetry?", np.allclose(zero_model.fc1.weight.data[0].numpy(), zero_model.fc1.weight.data[1].numpy()) == False)

    print("\n" + "=" * 70)
    print("TASK 4 (PART D): ACTIVATION EXPERIMENT (Sigmoid, Tanh, ReLU)")
    print("=" * 70)
    activations = ['sigmoid', 'tanh', 'relu']
    results = []

    for act_name in activations:
        torch.manual_seed(42)
        act_model = XORNet(activation=act_name)
        opt = optim.SGD(act_model.parameters(), lr=0.5)

        # Early gradient at step 1
        opt.zero_grad()
        l_out, _ = act_model(X)
        l = criterion(l_out, y)
        l.backward()
        early_norm = torch.norm(act_model.fc1.weight.grad, p=2).item()
        opt.step()

        # Train remaining steps
        for step in range(2, 3001):
            opt.zero_grad()
            l_out, _ = act_model(X)
            l = criterion(l_out, y)
            l.backward()
            opt.step()

        fin_l = l.item()
        fin_probs = torch.sigmoid(act_model(X)[0]).detach()
        fin_preds = (fin_probs >= 0.5).float()
        correct_4 = (fin_preds == y).sum().item() == 4
        results.append({
            'activation': act_name.capitalize(),
            'final_loss': fin_l,
            'correct': "Yes" if correct_4 else "No",
            'early_norm': early_norm,
            'probs': fin_probs.squeeze().tolist()
        })

    print(f"{'Hidden activation':<18} | {'Final loss':<12} | {'4/4 correct?':<12} | {'Early ||grad_W1||2':<18}")
    print("-" * 68)
    for r in results:
        print(f"{r['activation']:<18} | {r['final_loss']:<12.6f} | {r['correct']:<12} | {r['early_norm']:<18.6f}")

    print("\n" + "=" * 70)
    print("TASK 5: EXTENDED THREE-CLASS SENSOR PROBLEM")
    print("=" * 70)
    # Class 0: (0,0) inactive
    # Class 1: (0,1) or (1,0) disagreement
    # Class 2: (1,1) active
    y_multi = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    class MultiClassXORNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = nn.Tanh()
            self.fc2 = nn.Linear(2, 3)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(42)
    multi_model = MultiClassXORNet()
    multi_crit = nn.CrossEntropyLoss()
    multi_opt = optim.SGD(multi_model.parameters(), lr=0.5)

    print("Final weight matrix shape:", multi_model.fc2.weight.shape)

    for step in range(1, 3001):
        multi_opt.zero_grad()
        logits = multi_model(X)
        l = multi_crit(logits, y_multi)
        l.backward()
        multi_opt.step()

    print(f"Multi-class Final Loss: {l.item():.6f}")
    final_multi_logits = multi_model(X).detach()
    softmax_probs = torch.softmax(final_multi_logits, dim=-1)

    print("\nPredicted Class Probabilities:")
    for i in range(4):
        p_vec = softmax_probs[i].numpy()
        pred_class = np.argmax(p_vec)
        sum_p = np.sum(p_vec)
        print(f"Input: {X[i].numpy()} -> True Class: {y_multi[i].item()}, Probs: [{p_vec[0]:.4f}, {p_vec[1]:.4f}, {p_vec[2]:.4f}], Sum: {sum_p:.6f}, Pred: {pred_class}")

    # Diagnostic: Invariance to logit shifts
    logits_sample = final_multi_logits[0:1]
    p_orig = torch.softmax(logits_sample, dim=-1)
    p_shifted = torch.softmax(logits_sample + 100.0, dim=-1)
    print("\nOptional Diagnostic (Softmax Shift Invariance):")
    print("Original Logits:", logits_sample.numpy())
    print("Original Softmax:", p_orig.numpy())
    print("Shifted (+100) Softmax:", p_shifted.numpy())
    print("Max absolute difference:", torch.max(torch.abs(p_orig - p_shifted)).item())

if __name__ == '__main__':
    run_all()
