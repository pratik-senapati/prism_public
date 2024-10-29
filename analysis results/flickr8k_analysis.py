import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_flickr8k_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def analyze_flickr8k_results(results):
    cosine_similarities = []
    poly_similarities = []

    for item in results:
        cosine_similarities.append(item['cosine_similarity'])
        poly_similarities.append(item['poly_similarity'])
    
    avg_cosine_similarity = np.mean(cosine_similarities)
    avg_poly_similarity = np.mean(poly_similarities)

    std_cosine_similarity = np.std(cosine_similarities)
    std_poly_similarity = np.std(poly_similarities)

    median_cosine_similarity = np.median(cosine_similarities)
    median_poly_similarity = np.median(poly_similarities)

    mode_cosine_similarity = stats.mode(cosine_similarities, keepdims=True)[0][0]
    mode_poly_similarity = stats.mode(poly_similarities, keepdims=True)[0][0]

    print(f"Average cosine similarity: {avg_cosine_similarity:.4f}")
    print(f"Average polynomial similarity: {avg_poly_similarity:.4f}")
    print(f"Standard deviation of cosine similarity: {std_cosine_similarity:.4f}")
    print(f"Standard deviation of polynomial similarity: {std_poly_similarity:.4f}")
    print(f"Median cosine similarity: {median_cosine_similarity:.4f}")
    print(f"Median polynomial similarity: {median_poly_similarity:.4f}")
    print(f"Mode cosine similarity: {mode_cosine_similarity:.4f}")
    print(f"Mode polynomial similarity: {mode_poly_similarity:.4f}")

    # Plot histograms
    plt.figure()
    plt.hist(cosine_similarities, bins=20, alpha=0.7, color='blue')
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure()
    plt.hist(poly_similarities, bins=20, alpha=0.7, color='red')
    plt.title('Polynomial Similarity Distribution')
    plt.xlabel('Polynomial Similarity')
    plt.ylabel('Frequency')
    plt.show()

# Define the path to the results file
results_file = "flickr8k_test_results.json"

# Load the dataset
results = load_flickr8k_results(results_file)

# Analyze the results
analyze_flickr8k_results(results)