query_text = "Summarize the introduction"
query_vec = model.encode(query_text).tolist()

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vec,
    limit=5,
)

for res in results:
    print(f"\nScore: {res.score:.2f}")
    print(res.payload['text'])
