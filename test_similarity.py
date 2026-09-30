import numpy as np

from app.embeddings.embedder import EmbeddingModel


embedder = EmbeddingModel()


texts = [
    "Virtual Try-On helps consumers visualize products.",
    "Augmented reality allows shoppers to inspect products digitally.",
    "The experiment used an online questionnaire.",
]


embeddings = embedder.encode(texts)


def cosine_similarity(a, b):

    return np.dot(a, b)


print("\n" + "=" * 70)
print("SEMANTIC SIMILARITY TEST")
print("=" * 70)


for i in range(len(texts)):

    for j in range(i + 1, len(texts)):

        score = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        print(
            f"\nText {i + 1}:"
        )

        print(
            texts[i]
        )

        print(
            f"\nText {j + 1}:"
        )

        print(
            texts[j]
        )

        print(
            f"\nCosine similarity: "
            f"{score:.4f}"
        )

        print("-" * 70)