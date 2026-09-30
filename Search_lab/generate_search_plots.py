import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from run_search_experiments import a_star_search, ORIGINAL_WAREHOUSE

res = a_star_search(ORIGINAL_WAREHOUSE)
path = res['path']

rows = len(ORIGINAL_WAREHOUSE)
cols = len(ORIGINAL_WAREHOUSE[0])

grid_numeric = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        char = ORIGINAL_WAREHOUSE[r][c]
        if char == '#':
            grid_numeric[r, c] = 1 # obstacle
        elif char == 'S':
            grid_numeric[r, c] = 2 # start
        elif char == 'G':
            grid_numeric[r, c] = 3 # goal
        else:
            grid_numeric[r, c] = 0 # free

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
cmap = ListedColormap(['#f8fafc', '#2d3748', '#3182ce', '#38a169'])
ax.imshow(grid_numeric, cmap=cmap, origin='upper')

ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color='#cbd5e0', linestyle='-', linewidth=1)

# Annotate S and G
ax.text(1, 1, 'S', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.text(15, 7, 'G', ha='center', va='center', color='white', fontweight='bold', fontsize=11)

# Plot path
py, px = zip(*path)
ax.plot(px, py, color='#e53e3e', linewidth=2.5, marker='o', markersize=4, label='A* / BFS Path (40 steps)', zorder=4)

ax.set_title('Search Lab: Optimal 40-Step Serpentine Path (A* Search)', fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('search_warehouse_path.png')
plt.close()
print("Saved search_warehouse_path.png")

# Plot 2: BFS vs A* Search Efficiency Comparison
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
categories = ['Original Serpentine Map\n(64 Total Free Cells)', 'Open Warehouse Map\n(77 Total Free Cells)']
bfs_exp = [64, 59]
astar_exp = [64, 23]

x = np.arange(len(categories))
width = 0.32

rects1 = ax.bar(x - width/2, bfs_exp, width, label='BFS (Blind)', color='#4a5568')
rects2 = ax.bar(x + width/2, astar_exp, width, label='A* (Informed Manhattan)', color='#3182ce')

ax.set_ylabel('Number of States Expanded', fontsize=10, fontweight='bold')
ax.set_title('Task 5: Search Efficiency Comparison (States Expanded)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax.legend(frameon=True, fontsize=9)
ax.set_ylim(0, 75)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2b6cb0')

plt.tight_layout()
plt.savefig('search_bfs_vs_astar_comparison.png')
plt.close()
print("Saved search_bfs_vs_astar_comparison.png")
