from app.embeddings.embedder import EmbeddingModel


embedder = EmbeddingModel()


texts = [
    "Virtual Try-On helps consumers visualize products.",
    "Augmented reality allows shoppers to inspect products digitally.",
    "The experiment used an online questionnaire."
]


embeddings = embedder.encode(texts)


print("\n" + "=" * 70)
print("EMBEDDING TEST")
print("=" * 70)

print(
    f"Number of texts: {len(texts)}"
)

print(
    f"Embedding shape: {embeddings.shape}"
)

for i, embedding in enumerate(
    embeddings
):

    print(
        f"\nText {i + 1}"
    )

    print(
        f"Vector dimensions: "
        f"{len(embedding)}"
    )

    print(
        "First 10 values:"
    )

    print(
        embedding[:10]
    )