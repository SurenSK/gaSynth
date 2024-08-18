from typing import List
from collections import defaultdict
import random

class Sample:
    def __init__(self, scores: List[float]):
        self.scores = scores

def dominates(sample1: Sample, sample2: Sample) -> bool:
    """Check if one Sample dominates another (higher scores are better)."""
    at_least_one_strictly_better = False
    for i in range(len(sample1.scores)):
        if sample1.scores[i] > sample2.scores[i]:
            at_least_one_strictly_better = True
        elif sample1.scores[i] < sample2.scores[i]:
            return False
    return at_least_one_strictly_better

def sort_nondominated_samples(samples: List[Sample], k: int, first_front_only=False) -> List[List[Sample]]:
    """Sort the first k Samples into different nondomination levels."""
    if k == 0:
        return []

    map_score_sample = defaultdict(list)
    for sample in samples:
        map_score_sample[tuple(sample.scores)].append(sample)
    scores = list(map_score_sample.keys())

    current_front = []
    next_front = []
    dominating_scores = defaultdict(int)
    dominated_scores = defaultdict(list)

    for i, score_i in enumerate(scores):
        for score_j in scores[i + 1:]:
            if dominates(Sample(list(score_i)), Sample(list(score_j))):
                dominating_scores[score_j] += 1
                dominated_scores[score_i].append(score_j)
            elif dominates(Sample(list(score_j)), Sample(list(score_i))):
                dominating_scores[score_i] += 1
                dominated_scores[score_j].append(score_i)
        if dominating_scores[score_i] == 0:
            current_front.append(score_i)

    fronts = [[]]
    for score in current_front:
        fronts[-1].extend(map_score_sample[score])
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(samples), k)
        while pareto_sorted < N:
            fronts.append([])
            for score_p in current_front:
                for score_d in dominated_scores[score_p]:
                    dominating_scores[score_d] -= 1
                    if dominating_scores[score_d] == 0:
                        next_front.append(score_d)
                        pareto_sorted += len(map_score_sample[score_d])
                        fronts[-1].extend(map_score_sample[score_d])
            current_front = next_front
            next_front = []

    return fronts

from typing import List

def get_best_points(samples: List[Sample], k: int) -> List[Sample]:
    if k >= len(samples):
        return samples
    
    sel = []
    pareto_fronts = sort_nondominated_samples(samples)

    for front in pareto_fronts:
        if len(sel) + len(front) < k:
            sel.extend(front)
        else:
            needed = k - len(sel)
            # Subsample for the most diverse points within the current front
            selected_from_front = subsample_most_diverse(front, needed)
            sel.extend(selected_from_front)
            break  # We've collected enough samples

    return sel

def subsample_most_diverse(front: List[Sample], needed: int) -> List[Sample]:
    import random
    random.shuffle(front)
    return front[:needed]

# Test code
# random.seed(42)
samples = [Sample([random.uniform(0, 10), random.uniform(0, 10)]) for _ in range(20)]
pareto_fronts = sort_nondominated_samples(samples, k=len(samples))

for i, front in enumerate(pareto_fronts):
    print(f"Pareto Front {i + 1}:")
    for sample in front:
        print(f"  - Scores: {sample.scores}")

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_pareto_fronts(pareto_fronts):
    """Plots Pareto fronts with color gradient."""

    cmap = get_cmap('Oranges_r')  # Reverse Oranges colormap for bright to dark
    # cmap = get_cmap('prism')  # More contrast
    num_fronts = len(pareto_fronts)

    for i, front in enumerate(pareto_fronts):
        color = cmap(i / num_fronts)  # Color based on front index
        x = [sample.scores[0] for sample in front]
        y = [sample.scores[1] for sample in front]
        plt.scatter(x, y, color=color, label=f'Front {i + 1}', edgecolors='black')

    plt.xlabel('Score 1')
    plt.ylabel('Score 2')
    plt.title('Pareto Fronts')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

plot_pareto_fronts(pareto_fronts)