import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON data
file_path = 'flickr8k_test_results.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract relevant data
cosine_similarities = [item['cosine_similarity'] for item in data]
poly_similarities = [item['poly_similarity'] for item in data]
scaled_poly_similarities = [item['scaled_poly_similarity'] for item in data]
grades = [item['grade'] for item in data]

# Find the highest values
highest_cosine_similarity = max(data, key=lambda x: x['cosine_similarity'])
highest_poly_similarity = max(data, key=lambda x: x['poly_similarity'])
highest_scaled_poly_similarity = max(data, key=lambda x: x['scaled_poly_similarity'])

print("Highest Cosine Similarity:")
print(highest_cosine_similarity)
print("\nHighest Poly Similarity:")
print(highest_poly_similarity)
print("\nHighest Scaled Poly Similarity:")
print(highest_scaled_poly_similarity)

# Visualize the distribution of cosine_similarity
plt.figure(figsize=(10, 6))
sns.histplot(cosine_similarities, bins=30, kde=True)
plt.title('Distribution of Cosine Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()

# Visualize the distribution of poly_similarity
plt.figure(figsize=(10, 6))
sns.histplot(poly_similarities, bins=30, kde=True)
plt.title('Distribution of Poly Similarity')
plt.xlabel('Poly Similarity')
plt.ylabel('Frequency')
plt.show()

# Visualize the distribution of scaled_poly_similarity
plt.figure(figsize=(10, 6))
sns.histplot(scaled_poly_similarities, bins=30, kde=True)
plt.title('Distribution of Scaled Poly Similarity')
plt.xlabel('Scaled Poly Similarity')
plt.ylabel('Frequency')
plt.show()

# Visualize the grades
plt.figure(figsize=(10, 6))
sns.countplot(x=grades, order=['A', 'B', 'C', 'D', 'F'])
plt.title('Distribution of Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()

# Scatter plot of cosine_similarity vs. scaled_poly_similarity
plt.figure(figsize=(10, 6))
sns.scatterplot(x=cosine_similarities, y=scaled_poly_similarities, hue=grades)
plt.title('Cosine Similarity vs. Scaled Poly Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Scaled Poly Similarity')
plt.show()