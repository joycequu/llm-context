# from diskcache import Cache

# # Create a cache object with the default configuration
# cache = Cache()

# # Print the cache directory being used
# print("Default cache directory:", cache.directory)

import diskcache as dc
import os

cache_dir = '/home/gridsan/ywang5/hf/misc'
os.makedirs(cache_dir, exist_ok=True)
cache = dc.Cache(cache_dir)
cache.set('key', 'value')
print(cache.get('key'))
