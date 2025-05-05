# Shanni You, 2025/04/08
import chromadb

# This is a funtion to create a database
def dbCreate():

    client = chromadb.Client()  # maybe replaced with PersistentClient later
    test_collection = client.get_or_create_collection(name='test_collection', metadata={"hnsw:space": "cosine"})
    print(test_collection)

    # Adding Raw Documents: ids, embeddings, metadatas
    # Need to use embdding model or algorithm to replace it
    test_collection.add(
        documents = ["I know kung fu.", "There is no spoon."],
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ids = ['quote_1', 'quote_2'],
    )

    # Count items
    item_count = test_collection.count()
    print(f"Count of items in collection: {item_count}")

    # Retrieving items
    items = test_collection.get()
    print(items)

    # Taking a peek at a Collection
    test_collection.peek(limit = 5)

    # Usage of the collection
    # 1. Querying by a set of query embeddings
    results = test_collection.query(
        query_embeddings = [[0.1, 0.2, 0.3]],
        n_results = 1
    )
    print('test_result', results)










    # Delete a collection
    try: 
        client.delete_collection(name = "test_collection")
        print("Test_collection deleted.")
    except ValueError as e:
        print(f"Error: {e}")



def main():
    pass

if __name__ == '__main__':
    main()


