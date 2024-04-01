from dask_kubernetes.operator import KubeCluster
import time
from numpy import random
import pandas as pd
import dask

@dask.delayed
def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)

from dask_kubernetes import KubeCluster, make_pod_spec

# Define the pod spec for the worker
worker_pod_spec = make_pod_spec(
    image='registry.ai.dev1.intapp.com/workspace:dask9', # Specify your worker image
    extra_labels={'app': 'dask-worker'},       # Custom label for worker pods
    # ... other worker configuration ...
)

# Define the pod spec for the scheduler
scheduler_pod_spec = make_pod_spec(
    image='registry.ai.dev1.intapp.com/scheduler:4.0', # Specify your scheduler image
    extra_labels={'app': 'dask-scheduler'},       # Custom label for scheduler pod
    # ... other scheduler configuration ...
)

# Create the cluster object with separate specs for the scheduler and worker
cluster = KubeCluster(
    pod_template=worker_pod_spec, 
    scheduler_pod_template=scheduler_pod_spec
)



# cluster = KubeCluster(
#     name="dask-cluster9",
#     image="registry.ai.dev1.intapp.com/workspace:dask9",
#     resources={"requests": {"memory": "2Gi"}, "limits": {"memory": "64Gi"}},
# )
cluster.scale(1)
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