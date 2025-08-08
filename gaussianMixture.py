import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture

# 1. Load digits data
digits = load_digits()
X = digits.data
y = digits.target

# print(X, y)

# Train a separate GMM for each digit
digit_models = {}  # Dictionary with 10 trained models
n_components = 15  # Number of components per digit model

digit_models = {digit: GaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(X[y == digit]) for digit in range(10)}
# for digit in range(10):
#     X_digit = X[y == digit]  # all images of this digit
#     gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
#     gmm.fit(X_digit)
#     digit_models[digit] = gmm

# Generate a new image for each digit
generated_digits = [digit_models[digit].sample(1)[0].reshape(8,8) for digit in range(10)]
# for digit in range(10):
#     gmm = digit_models[digit]  # corresponding GMM
#     sample, _ = gmm.sample(1)  # generate image
#     generated_digits.append(sample.reshape(8, 8))  # reshape image to 8x8

# Visualization
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i, image in enumerate(generated_digits):
    axes[i].imshow(image, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(str(i))

plt.suptitle("Generated digits from their own GMM", fontsize=14)
plt.tight_layout()
plt.show()





