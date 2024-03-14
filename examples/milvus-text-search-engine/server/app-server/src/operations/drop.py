import sys
sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER

def do_drop(milvus_cli, mysql_cli, table_name=DEFAULT_TABLE):
    try:
        if not milvus_cli.has_collection(table_name):
            return "Collection does not exist"
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with drop table: {e}")
        sys.exit(1)
