import time
from dask_kubernetes import HelmCluster
from numpy import random
import pandas as pd
import dask

@dask.delayed
def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)

# Create the cluster object with separate specs for the scheduler and worker
cluster = HelmCluster('dask')


# cluster.scale(1)
client = cluster.get_client()


input_params = pd.DataFrame(random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])
input_params.head()

lazy_results = []
for parameters in input_params.values:
    # lazy_result = dask.delayed(costly_simulation)(parameters)
    lazy_result = costly_simulation(parameters)
    lazy_results.append(lazy_result)

futures = dask.persist(*lazy_results)  # trigger computation in the background

results = dask.compute(*futures)
print(results[:5])
cluster.close()