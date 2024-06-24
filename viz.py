import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def extract_scores(file_path):
    all_scores = []
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            if 'sScores:' in line:
                scores = eval(line.split('sScores:')[1].strip())
                id = ["a", "b", "c", "d", "e", "f"]
                for j,score in enumerate(scores):
                    all_scores.append((str(i)+id[j], tuple(score)))
                i += 1
    
    print(f"Total scores extracted: {len(all_scores)}")
    print(f"First 5 extracted scores: {all_scores[:5]}")
    print(f"Last 5 extracted scores: {all_scores[-5:]}")
    
    return all_scores

def plot_scores(scores_with_ops):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    ops, scores = zip(*scores_with_ops)
    x, y, z = zip(*scores)

    num_points = len(scores)
    op_numbers = [int(op[:-1]) for op in ops]  # Extract numeric part of the op
    unique_ops = sorted(set(op_numbers))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ops)))
    color_dict = dict(zip(unique_ops, colors))

    # Add small jitter to prevent perfect overlaps
    jitter = 0.002
    x = np.array(x) + np.random.uniform(-jitter, jitter, num_points)
    y = np.array(y) + np.random.uniform(-jitter, jitter, num_points)
    z = np.array(z) + np.random.uniform(-jitter, jitter, num_points)

    for op in unique_ops:
        mask = [int(o[:-1]) == op for o in ops]
        scatter = ax.scatter(
            [x[i] for i in range(num_points) if mask[i]],
            [y[i] for i in range(num_points) if mask[i]],
            [z[i] for i in range(num_points) if mask[i]],
            c=[color_dict[op]],
            s=50,
            label=f'Op {op}',
            alpha=0.7,
            edgecolors='black'
        )

    ax.set_xlabel('Length')
    ax.set_ylabel('Deceptiveness')
    ax.set_zlabel('Completeness')
    ax.set_title(f'3D Score Plot ({num_points} points)')

    # Set axis limits to 0-1 with some padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)

    # Add gridlines
    ax.grid(True)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"Op: {ops[ind['ind'][0]]}\nScores: {scores[ind['ind'][0]]}"
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.tight_layout()
    plt.show()

# Use the functions
try:
    scores_with_ops = extract_scores('log2.txt')
    if scores_with_ops:
        plot_scores(scores_with_ops)
    else:
        print("No scores found in the log file.")
except FileNotFoundError:
    print("Error: 'log2.txt' not found in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")