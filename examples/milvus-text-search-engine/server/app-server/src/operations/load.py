import sys, time
import numpy as np
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER

# Get the vector of search
def extract_features(path, model):
    try:
        data = pd.read_csv(path)
        title_data = data['title'].tolist()
        text_data = data['text'].tolist()
        sentence_embeddings = model.sentence_encode(text_data, is_load=True)
        return title_data, text_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)

# format data for submission to mmysql
def format_data_mysql(ids, title_data, text_data):
    # combine the id of the vector and question data into list of tuples
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), title_data[i], text_data[i])
        data.append(value)
    return data

# Import vectors to milvus + create local lookup table
def do_load(embedding_model, milvus_client, mysql_client, data_path, collection_name=DEFAULT_TABLE):
    start = time.perf_counter()
    title_data, text_data, sentence_embeddings = extract_features(data_path, embedding_model)
    end = time.perf_counter()
    
    start_db = time.perf_counter()
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    milvus_client.create_index(collection_name)
    mysql_client.create_mysql_table(collection_name)
    mysql_client.load_data_to_mysql(collection_name, format_data_mysql(ids, title_data, text_data))
    end_db = time.perf_counter()
    
    return len(ids), end - start, end_db - start_db
