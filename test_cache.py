# from diskcache import Cache

# # Create a cache object with the default configuration
# cache = Cache()

# # Print the cache directory being used
# print("Default cache directory:", cache.directory)

# import diskcache as dc
# import os, logging

# logging.basicConfig(level=logging.DEBUG)

# cache_dir = '/state/partition1/user/$USER'
# os.makedirs(cache_dir, exist_ok=True)
# cache = dc.Cache(cache_dir)
# cache.set('key', 'value')
# print(cache.get('key'))

# import json
# import os
# from vllm import LLM

# def main():
#     cls_path = '/state/partition1/user/$USER'
#     if not os.path.exists(cls_path):
#         os.makedirs(cls_path)
    
#     try:
#         cls = LLM(model=cls_path, tensor_parallel_size=1)
#         print('LLM initialized successfully')
#     except Exception as e:
#         print('Error initializing LLM:', e)

# if __name__ == "__main__":
#     main()

import sqlite3
import os

# Use your custom directory
db_path = os.path.join("/state/partition1/user/$USER", "test_db.sqlite")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)")
    cursor.execute("INSERT INTO test (data) VALUES ('test data')")
    conn.commit()
    cursor.execute("SELECT * FROM test")
    print(cursor.fetchall())
except sqlite3.OperationalError as e:
    print(f"SQLite error: {e}")
finally:
    conn.close()
    os.remove(db_path)  # Clean up
