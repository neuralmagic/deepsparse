import sys

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


def do_count(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error( f"Error with count table {e}")
        sys.exit(1)
