from dask.distributed import Client, get_worker

# Create a Dask cluster
client = Client('tcp://dask-scheduler:8786')

# Submit a task to the cluster
def my_task(x):
    return x + 1

future = client.submit(my_task, 1)

# Get the result of the task
result = future.result()

# Print the result
print(result)
