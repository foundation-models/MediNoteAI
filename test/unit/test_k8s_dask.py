from dask_kubernetes.operator import KubeCluster
import time
from numpy import random
import pandas as pd
import dask

@dask.delayed
def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)


cluster = KubeCluster(
    name="dask-cluster",
    image="ghcr.io/dask/dask:latest",
    resources={"requests": {"memory": "2Gi"}, "limits": {"memory": "64Gi"}},
)
cluster.scale(3)
client = cluster.get_client()


input_params = pd.DataFrame(random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])
input_params.head()

# results = []
# for parameters in input_params.values[:10]:
#     result = costly_simulation(parameters)
#     results.append(result)

import dask
lazy_results = []
for parameters in input_params.values:
    # lazy_result = dask.delayed(costly_simulation)(parameters)
    lazy_result = costly_simulation(parameters)
    lazy_results.append(lazy_result)

futures = dask.persist(*lazy_results)  # trigger computation in the background

results = dask.compute(*futures)
print(results[:5])
cluster.close()