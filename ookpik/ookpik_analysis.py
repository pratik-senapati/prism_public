import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_ookpik_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    accurate_cosine_similarities = []
    accurate_poly_similarities = []
    inaccurate_cosine_similarities = []
    inaccurate_poly_similarities = []

    for item in results:
        context_label = item['context_label']
        if context_label == 1:
            inaccurate_cosine_similarities.append(item['cosine_similarity1'])
            inaccurate_cosine_similarities.append(item['cosine_similarity2'])
            inaccurate_poly_similarities.append(item['poly_similarity1'])
            inaccurate_poly_similarities.append(item['poly_similarity2'])
        elif context_label == 0:
            accurate_cosine_similarities.append(item['cosine_similarity1'])
            accurate_cosine_similarities.append(item['cosine_similarity2'])
            accurate_poly_similarities.append(item['poly_similarity1'])
            accurate_poly_similarities.append(item['poly_similarity2'])
    
    total_samples = len(results)
    accurate_count = len(accurate_cosine_similarities) // 2
    inaccurate_count = len(inaccurate_cosine_similarities) // 2

    avg_accurate_cosine_similarity = np.mean(accurate_cosine_similarities)
    avg_inaccurate_cosine_similarity = np.mean(inaccurate_cosine_similarities)
    avg_accurate_poly_similarity = np.mean(accurate_poly_similarities)
    avg_inaccurate_poly_similarity = np.mean(inaccurate_poly_similarities)

    std_accurate_cosine_similarity = np.std(accurate_cosine_similarities)
    std_inaccurate_cosine_similarity = np.std(inaccurate_cosine_similarities)
    std_accurate_poly_similarity = np.std(accurate_poly_similarities)
    std_inaccurate_poly_similarity = np.std(inaccurate_poly_similarities)

    median_accurate_cosine_similarity = np.median(accurate_cosine_similarities)
    median_inaccurate_cosine_similarity = np.median(inaccurate_cosine_similarities)
    median_accurate_poly_similarity = np.median(accurate_poly_similarities)
    median_inaccurate_poly_similarity = np.median(inaccurate_poly_similarities)

    mode_accurate_cosine_similarity = stats.mode(accurate_cosine_similarities, keepdims=True)[0][0]
    mode_inaccurate_cosine_similarity = stats.mode(inaccurate_cosine_similarities, keepdims=True)[0][0]
    mode_accurate_poly_similarity = stats.mode(accurate_poly_similarities, keepdims=True)[0][0]
    mode_inaccurate_poly_similarity = stats.mode(inaccurate_poly_similarities, keepdims=True)[0][0]

    print(f"Total samples: {total_samples}")
    print(f"Accurate captions: {accurate_count} ({(accurate_count / total_samples) * 100:.2f}%)")
    print(f"Inaccurate captions: {inaccurate_count} ({(inaccurate_count / total_samples) * 100:.2f}%)")
    print(f"Average cosine similarity (accurate): {avg_accurate_cosine_similarity:.4f}")
    print(f"Average cosine similarity (inaccurate): {avg_inaccurate_cosine_similarity:.4f}")
    print(f"Average polynomial similarity (accurate): {avg_accurate_poly_similarity:.4f}")
    print(f"Average polynomial similarity (inaccurate): {avg_inaccurate_poly_similarity:.4f}")
    print(f"Standard deviation of cosine similarity (accurate): {std_accurate_cosine_similarity:.4f}")
    print(f"Standard deviation of cosine similarity (inaccurate): {std_inaccurate_cosine_similarity:.4f}")
    print(f"Standard deviation of polynomial similarity (accurate): {std_accurate_poly_similarity:.4f}")
    print(f"Standard deviation of polynomial similarity (inaccurate): {std_inaccurate_poly_similarity:.4f}")
    print(f"Median cosine similarity (accurate): {median_accurate_cosine_similarity:.4f}")
    print(f"Median cosine similarity (inaccurate): {median_inaccurate_cosine_similarity:.4f}")
    print(f"Median polynomial similarity (accurate): {median_accurate_poly_similarity:.4f}")
    print(f"Median polynomial similarity (inaccurate): {median_inaccurate_poly_similarity:.4f}")
    print(f"Mode cosine similarity (accurate): {mode_accurate_cosine_similarity:.4f}")
    print(f"Mode cosine similarity (inaccurate): {mode_inaccurate_cosine_similarity:.4f}")
    print(f"Mode polynomial similarity (accurate): {mode_accurate_poly_similarity:.4f}")
    print(f"Mode polynomial similarity (inaccurate): {mode_inaccurate_poly_similarity:.4f}")

    # Plot histograms in separate windows
    plt.figure()
    plt.hist(accurate_cosine_similarities, bins=20, alpha=0.7, color='blue')
    plt.title('Cosine Similarity Distribution (Accurate)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure()
    plt.hist(inaccurate_cosine_similarities, bins=20, alpha=0.7, color='red')
    plt.title('Cosine Similarity Distribution (Inaccurate)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure()
    plt.hist(accurate_poly_similarities, bins=20, alpha=0.7, color='blue')
    plt.title('Polynomial Similarity Distribution (Accurate)')
    plt.xlabel('Polynomial Similarity')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure()
    plt.hist(inaccurate_poly_similarities, bins=20, alpha=0.7, color='red')
    plt.title('Polynomial Similarity Distribution (Inaccurate)')
    plt.xlabel('Polynomial Similarity')
    plt.ylabel('Frequency')
    plt.show()

# Define the path to the results file
results_file = "ookpik_test_results.json"

# Analyze the results
analyze_ookpik_results(results_file)