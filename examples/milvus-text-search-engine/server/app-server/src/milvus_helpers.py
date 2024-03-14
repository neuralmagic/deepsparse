import sys
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from config import MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, INDEX_TYPE, NLIST, NPROBE
from logs import LOGGER

class MilvusHelper:
    """
      class
    """
    def __init__(self, milvus_host):
        try:
            self.collection = None
            connections.connect(host=milvus_host, port=MILVUS_PORT)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{milvus_host,} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(f"There has no collection named:{collection_name}")
        except Exception as e:
            LOGGER.error(f"Error: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            status = utility.has_collection(collection_name)
            return status
        except Exception as e:
            LOGGER.error(f"Failed to check collection: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            if not self.has_collection(collection_name):
                field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True, auto_id=True)
                field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector", dim=VECTOR_DIMENSION, is_primary=False)
                schema = CollectionSchema(fields=[field1,field2], description="collection description")
                self.collection = Collection(name=collection_name, schema=schema)
                LOGGER.debug(f"Create Milvus collection: {self.collection}")
            return "Successfully created collection"
        except Exception as e:
            LOGGER.error(f"Failed to create collection: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors):
        # Batch insert vectors to milvus collection
        try:
            self.create_collection(collection_name)
            self.collection = Collection(name=collection_name)
            data = [vectors]
            mr = self.collection.insert(data)
            ids =  mr.primary_keys
            self.collection.load()
            LOGGER.debug(f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data into Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        # Create IVF_SQ8 index on milvus collection
        try:
            self.set_collection(collection_name)
            default_index = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": {"nlist": NLIST}}
            status = self.collection.create_index(field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                   f"Successfully create index in collection:{collection_name} with param:{default_index}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
         # Delete Milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "Successfully dropped collection"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        # Search vector in milvus collection
        try:
            self.set_collection(collection_name)
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": NPROBE}}
            res=self.collection.search(vectors, anns_field="embedding", param=search_params, limit=top_k)
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
         # Get the number of milvus collection
        try:
            self.set_collection(collection_name)
            num = self.collection.num_entities
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)

