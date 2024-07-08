from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_distances(obvious_questions, new_questions):
    # Encode all questions
    obvious_embeddings = embedding_model.encode(obvious_questions)
    new_embeddings = embedding_model.encode(new_questions)
    
    # Calculate cosine distances
    distances = np.zeros((len(obvious_questions), len(new_questions)))
    for i, obvious_emb in enumerate(obvious_embeddings):
        for j, new_emb in enumerate(new_embeddings):
            distances[i, j] = cosine(obvious_emb, new_emb)
    
    return distances

# Predefined obvious questions
obvious_questions = [
    "How do I mix the ingredients for a cake?",
    "What temperature should I bake a cake at?",
    "How long does it take to bake a cake?",
    "What ingredients do I need to bake a cake?",
    "How do I ensure the cake rises properly?"
]

# New questions to compare
new_questions = [
    "How do I mix the ingredients for a cake?",
    "What temperature should I bake a cake at?",
    "How long does it take to bake a cake?",
    "What ingredients do I need to bake a cake?",
    "How do I ensure the cake rises properly?",
    "How should I whip the cream for the cake?",
    "What type of flour is best for a fluffy cake?",
    "What's the best method to combine sugar and butter for a creamy texture?",
    "Is there a recommended duration for whipping eggs to incorporate air?",
    "Can you suggest some essential items for someone starting with home baking?",
    "What are the benefits of using baking powder in a recipe?",
    "How do I get a moist texture when preparing desserts?",
    "How do I make a chocolate cake?",
    "What's the difference between baking soda and baking powder?"
]

# Calculate distances
distances = calculate_cosine_distances(obvious_questions, new_questions)
print("Cosine distances for each new question:")
for j, new_q in enumerate(new_questions):
    distances_to_obvious = [distances[i, j] for i in range(len(obvious_questions))]
    distances_formatted = ', '.join(f"{dist:.4f}" for dist in distances_to_obvious)
    avgD = sum(distances_to_obvious)/len(distances_to_obvious)
    minD = min(distances_to_obvious)
    print(f'{minD:.2f} : {avgD:.2f} : {avgD+minD:.2f} : {new_q}')
# Calculate distances
