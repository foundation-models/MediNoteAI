from dask_kubernetes.operator import KubeCluster
import time
from numpy import random
import pandas as pd
import dask

@dask.delayed
def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)


# Define the pod spec for the worker
# worker_pod_spec = make_pod_spec(
#     image='foundationmodels/workspace', # Specify your worker image
#     labels={'app': 'dask-worker'},       # Custom label for worker pods
#     # ... other worker configuration ...
# )

# # Define the pod spec for the scheduler
# scheduler_pod_spec = make_pod_spec(
#     image='rfoundationmodels/scheduler', # Specify your scheduler image
#     labels={'app': 'dask-scheduler'},       # Custom label for scheduler pod
#     # ... other scheduler configuration ...
# )

# # Create the cluster object with separate specs for the scheduler and worker
# cluster = KubeCluster(
#     pod_template=worker_pod_spec, 
#     scheduler_pod_template=scheduler_pod_spec
# )






    # worker_command: List[str] | str
    #     The command to use when starting the worker.
    #     If command consists of multiple words it should be passed as a list of strings.
    #     Defaults to ``"dask-worker"``.
    # port_forward_cluster_ip: bool (optional)
    #     If the chart uses ClusterIP type services, forward the
    #     ports locally. If you are running it locally it should
    #     be the port you are forwarding to ``<port>``.
    # create_mode: CreateMode (optional)
    #     How to handle cluster creation if the cluster resource already exists.
    #     Default behavior is to create a new cluster if one with that name
    #     doesn't exist, or connect to an existing one if it does.
    #     You can also set ``CreateMode.CREATE_ONLY`` to raise an exception if a cluster
    #     with that name already exists. Or ``CreateMode.CONNECT_ONLY`` to raise an exception
    #     if a cluster with that name doesn't exist.
    # shutdown_on_close: bool (optional)
    #     Whether or not to delete the cluster resource when this object is closed.
    #     Defaults to ``True`` when creating a cluster and ``False`` when connecting to an existing one.
    # idle_timeout: int (optional)
    #     If set Kubernetes will delete the cluster automatically if the scheduler is idle for longer than
    #     this timeout in seconds.
    # resource_timeout: int (optional)
    #     Time in seconds to wait for Kubernetes resources to enter their expected state.
    #     Example: If the ``DaskCluster`` resource that gets created isn't moved into a known ``status.phase``
    #     by the controller then it is likely the controller isn't running or is malfunctioning and we time
    #     out and clean up with a useful error.
    #     Example 2: If the scheduler Pod enters a ``CrashBackoffLoop`` state for longer than this timeout we
    #     give up with a useful error.
    #     Defaults to ``60`` seconds.
    # scheduler_service_type: str (optional)
    #     Kubernetes service type to use for the scheduler. Defaults to ``ClusterIP``.
    # jupyter: bool (optional)
    #     Start Jupyter on the scheduler node.
    # custom_cluster_spec: str | dict (optional)
    #     Path to a YAML manifest or a dictionary representation of a ``DaskCluster`` resource object which will be
    #     used to create the cluster instead of generating one from the other keyword arguments.
    # scheduler_forward_port: int (optional)
    #     The port to use when forwarding the scheduler dashboard. Will utilize a random port by default
    # quiet: bool
    #     If enabled, suppress all printed output.
    #     Defaults to ``False``.



cluster = KubeCluster(
    name="dask-cluster24",
    namespace="ai",
    n_workers=3,
    image="registry.ai.dev1.intapp.com/workspace:dask11",
    resources={"requests": {"memory": "2Gi"}, "limits": {"memory": "64Gi"}},
    env={"PYTHONPATH": "/home/agent/workspace/MediNoteAI/src:/home/agent/workspace/MediNoteAI/tests"},
)
# cluster.scale(1)
client = cluster.get_client()

print("sleeping ...")
time.sleep(300)
# input_params = pd.DataFrame(random.random(size=(500, 4)),
#                             columns=['param_a', 'param_b', 'param_c', 'param_d'])
# input_params.head()

# results = []
# for parameters in input_params.values[:10]:
#     result = costly_simulation(parameters)
#     results.append(result)

lazy_results = []
for parameters in input_params.values:
    # lazy_result = dask.delayed(costly_simulation)(parameters)
    lazy_result = costly_simulation(parameters)
    lazy_results.append(lazy_result)

# futures = dask.persist(*lazy_results)  # trigger computation in the background

# results = dask.compute(*futures)
# print(results[:5])
# cluster.close()