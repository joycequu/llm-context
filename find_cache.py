from diskcache import Cache

# Create a cache object with the default configuration
cache = Cache()

# Print the cache directory being used
print("Default cache directory:", cache.directory)