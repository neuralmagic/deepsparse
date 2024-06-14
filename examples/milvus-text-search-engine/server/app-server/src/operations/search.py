import sys
import numpy as np

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER

def search_milvus(query_sentence, model, milvus_cli, mysql_cli, table_name=DEFAULT_TABLE):
    try:
        vectors = model.sentence_encode([query_sentence])
        results = milvus_cli.search_vectors(table_name, vectors, TOP_K)
        vids = [str(x.id) for x in results[0]]
        ids, title, text = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in results[0]]
        return ids, title, text, distances
    
    except Exception as e:
        LOGGER.error(f" Error with search : {e}")
        sys.exit(1)