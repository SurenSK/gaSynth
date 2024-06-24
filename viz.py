import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def extract_scores_and_ops(log_text):
    pattern = r'Op (\d+).*?sScores:\[(\[[\d., ]+\]),\s*(\[[\d., ]+\])\]'
    matches = re.findall(pattern, log_text, re.DOTALL)
    
    print("Matches found:", len(matches))
    for i, match in enumerate(matches[:5]):  # Print first 5 matches
        print(f"Match {i+1}: Op {match[0]}, Scores: {match[1]}, {match[2]}")
    
    all_scores = []
    for op, score1, score2 in matches:
        all_scores.extend([
            (int(op), tuple(map(float, eval(score1)))),
            (int(op), tuple(map(float, eval(score2))))
        ])
    
    print("Total scores extracted:", len(all_scores))
    print("First 5 extracted scores:", all_scores[:5])
    
    return all_scores

def plot_scores(scores_with_ops):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ops, scores = zip(*scores_with_ops)
    x, y, z = zip(*scores)

    num_points = len(scores)
    colors = plt.cm.plasma(np.linspace(0, 1, num_points))

    scatter = ax.scatter(x, y, z, c=colors, s=50)

    ax.set_xlabel('Length')
    ax.set_ylabel('Deceptiveness')
    ax.set_zlabel('Completeness')
    ax.set_title('3D Score Plot')

    plt.colorbar(scatter, label='Sample order (Black to Orange)')

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

    plt.show()

# Load log data from file
try:
    with open('log.txt', 'r') as file:
        log_text = file.read()
except FileNotFoundError:
    print("Error: 'log.txt' not found in the current directory.")
    exit(1)

scores_with_ops = extract_scores_and_ops(log_text)
if scores_with_ops:
    plot_scores(scores_with_ops)
else:
    print("No scores found in the log file.")