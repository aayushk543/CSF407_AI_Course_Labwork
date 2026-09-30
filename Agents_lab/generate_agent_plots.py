import matplotlib.pyplot as plt
import numpy as np

# Plot 1: Warehouse Grid & Path Visualization
WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# Path from BFS:
path = [
    (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (2, 6), (2, 7),
    (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14),
    (1, 15), (1, 16), (1, 17), (1, 18), (1, 19)
]

rows = len(WAREHOUSE_MAP)
cols = len(WAREHOUSE_MAP[0])

grid_numeric = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        char = WAREHOUSE_MAP[r][c]
        if char == '#':
            grid_numeric[r, c] = 1 # obstacle
        elif char == 'S':
            grid_numeric[r, c] = 2 # start
        elif char == 'G':
            grid_numeric[r, c] = 3 # goal
        else:
            grid_numeric[r, c] = 0 # free

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

# Colors: 0: free (white), 1: obstacle (dark gray), 2: start (blue), 3: goal (green)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#f8fafc', '#2d3748', '#3182ce', '#38a169'])

ax.imshow(grid_numeric, cmap=cmap, origin='upper')

# Major ticks
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color='#cbd5e0', linestyle='-', linewidth=1)

# Annotate Start and Goal
ax.text(1, 1, 'S', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
ax.text(19, 1, 'G', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

# Plot Path line and points
py, px = zip(*path)
ax.plot(px, py, color='#e53e3e', linewidth=3, marker='o', markersize=6, label='Planned Path (20 steps)', zorder=4)

ax.set_title('Warehouse Navigation: Goal-Based Agent Planned Trajectory (BFS)', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('warehouse_path_visualization.png')
plt.close()
print("Saved warehouse_path_visualization.png")

# Plot 2: Goal-Based Agent Architecture Diagram
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.axis('off')

boxes = {
    'Environment': (0.1, 0.45, 0.22, 0.2, '#edf2f7', '#4a5568', 'Environment\n(Warehouse Grid)'),
    'Sensors': (0.42, 0.72, 0.22, 0.16, '#e2e8f0', '#2d3748', 'Sensors\n(Grid Percepts)'),
    'State': (0.42, 0.45, 0.22, 0.2, '#ebf8ff', '#2b6cb0', 'Internal State\n(Current Pos & Map)'),
    'Goal': (0.75, 0.72, 0.22, 0.16, '#feebc8', '#c05621', 'Goal Formulation\n(Reach Cell G)'),
    'Decision': (0.75, 0.45, 0.22, 0.2, '#feebc8', '#dd6b20', 'Decision Maker\n(BFS Search Engine)'),
    'Actuators': (0.42, 0.15, 0.22, 0.16, '#e2e8f0', '#2d3748', 'Actuators\n(Motor Stepper)'),
}

for name, (x, y, w, h, bg, border, text) in boxes.items():
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=border, linewidth=2, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color=border, transform=ax.transAxes, zorder=3)

# Connect with arrows
arrowprops = dict(arrowstyle="->", lw=2, color="#4a5568")
# Env -> Sensors
ax.annotate("", xy=(0.42, 0.8), xytext=(0.32, 0.6), arrowprops=arrowprops)
# Sensors -> State
ax.annotate("", xy=(0.53, 0.65), xytext=(0.53, 0.72), arrowprops=arrowprops)
# State -> Decision
ax.annotate("", xy=(0.75, 0.55), xytext=(0.64, 0.55), arrowprops=arrowprops)
# Goal -> Decision
ax.annotate("", xy=(0.86, 0.65), xytext=(0.86, 0.72), arrowprops=arrowprops)
# Decision -> Actuators
ax.annotate("", xy=(0.64, 0.23), xytext=(0.75, 0.45), arrowprops=arrowprops)
# Actuators -> Env
ax.annotate("", xy=(0.32, 0.48), xytext=(0.42, 0.23), arrowprops=arrowprops)

ax.set_title('Task 2: Goal-Based Agent Architecture (Russell & Norvig Model)', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('agent_architecture_diagram.png')
plt.close()
print("Saved agent_architecture_diagram.png")
