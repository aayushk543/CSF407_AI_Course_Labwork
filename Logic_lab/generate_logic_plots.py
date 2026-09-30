import matplotlib.pyplot as plt

# Figure 1: Planning Architecture: Logic + Search
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.axis('off')

boxes = {
    'State': (0.35, 0.82, 0.3, 0.12, '#ebf8ff', '#2b6cb0', 'Current State S\n{At(Robot,A), At(Package,A)}'),
    'Logic_Pre': (0.35, 0.62, 0.3, 0.12, '#feebc8', '#c05621', 'Logical Deduction\nS |= Preconditions(a)?'),
    'Applicable': (0.35, 0.42, 0.3, 0.12, '#e2e8f0', '#2d3748', 'Applicable Actions A_valid\n{PickUp(Package,A), Move(A,B)}'),
    'Logic_Eff': (0.35, 0.22, 0.3, 0.12, '#e6fffa', '#234e52', 'Successor Generation\nS\' = (S \\ Neg) U Pos'),
    'Search': (0.75, 0.42, 0.22, 0.32, '#faf5ff', '#553c9a', 'Search Engine (BFS)\nFrontier Queue\nCycle Detection\nExplores Alternat.'),
    'Goal': (0.35, 0.02, 0.3, 0.12, '#f0fff4', '#22543d', 'Goal Test\nS\' |= G {At(Package, C)}?'),
}

for name, (x, y, w, h, bg, border, text) in boxes.items():
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=border, linewidth=1.8, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=border, transform=ax.transAxes, zorder=3)

arrowprops = dict(arrowstyle="->", lw=1.8, color="#4a5568")
# State -> Logic_Pre
ax.annotate("", xy=(0.5, 0.74), xytext=(0.5, 0.82), arrowprops=arrowprops)
# Logic_Pre -> Applicable
ax.annotate("", xy=(0.5, 0.54), xytext=(0.5, 0.62), arrowprops=arrowprops)
# Applicable -> Logic_Eff
ax.annotate("", xy=(0.5, 0.34), xytext=(0.5, 0.42), arrowprops=arrowprops)
# Logic_Eff -> Goal
ax.annotate("", xy=(0.5, 0.14), xytext=(0.5, 0.22), arrowprops=arrowprops)
# Logic_Eff -> Search
ax.annotate("", xy=(0.75, 0.5), xytext=(0.65, 0.28), arrowprops=arrowprops)
# Search -> State (loop back)
ax.annotate("", xy=(0.65, 0.88), xytext=(0.75, 0.65), arrowprops=arrowprops)

ax.set_title('Task 4: Logical Reasoning + State-Space Search = Planning', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('planning_architecture_diagram.png')
plt.close()
print("Saved planning_architecture_diagram.png")

# Figure 2: Plan Trajectory State Transition Graph
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.axis('off')

states_data = [
    ("S0: Initial State", "{At(Robot,A),\nAt(Package,A)}", 0.05),
    ("S1: Picked Up", "{At(Robot,A),\nHolding(Pkg)}", 0.28),
    ("S2: Moved to B", "{At(Robot,B),\nHolding(Pkg)}", 0.51),
    ("S3: Moved to C", "{At(Robot,C),\nHolding(Pkg)}", 0.74),
    ("S4: Goal State", "{At(Robot,C),\nAt(Package,C)}", 0.97),
]

actions_labels = [
    "PickUp(Pkg, A)",
    "Move(A, B)",
    "Move(B, C)",
    "Drop(Pkg, C)",
]

for idx, (title, facts, x_pos) in enumerate(states_data):
    # draw box
    rect = plt.Rectangle((x_pos - 0.08, 0.3), 0.14, 0.45, facecolor='#ebf8ff' if idx < 4 else '#f0fff4',
                         edgecolor='#2b6cb0' if idx < 4 else '#276749', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_pos - 0.01, 0.63, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2d3748')
    ax.text(x_pos - 0.01, 0.45, facts, ha='center', va='center', fontsize=7, color='#4a5568')

    if idx < 4:
        # draw arrow to next
        next_x = states_data[idx+1][2]
        ax.annotate("", xy=(next_x - 0.08, 0.52), xytext=(x_pos + 0.06, 0.52),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#e53e3e"))
        ax.text((x_pos + next_x)/2 - 0.01, 0.59, actions_labels[idx], ha='center', va='bottom', fontsize=7, fontweight='bold', color='#c53030')

ax.set_xlim(-0.02, 1.05)
ax.set_ylim(0.1, 0.9)
ax.set_title('Task 1: Valid State Transition Sequence for Warehouse Goal Achievement', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('state_transition_plan_trace.png')
plt.close()
print("Saved state_transition_plan_trace.png")
