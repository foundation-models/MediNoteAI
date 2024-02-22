import redis

# Connect to Redis server
redis_client = redis.Redis(host='redis', port=6379, db=1)

# Prefix to search for
prefix = "test4_2024-02-22 16:17:29.266583_*"

# Fetch all keys starting with the prefix
keys = redis_client.keys(prefix)

# Sort the keys in descending order
sorted_keys = sorted(keys, reverse=True)

print(len(sorted_keys))
# Print the sorted keys
for key in sorted_keys:
    print(key.decode('utf-8'))
    