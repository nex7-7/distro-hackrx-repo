import weaviate

# Connect to local Weaviate instance
client = weaviate.connect_to_local(host="weaviate", port=8080)

# Get schema object
schema = client.collections.list_all()

# Delete all collections (formerly "classes")
for collection in schema:
    print(f"Deleting collection: {collection}")
    client.collections.delete(collection)

print("All collections deleted. Weaviate is now empty.")

# Properly close connection
client.close()
