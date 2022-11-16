from deployments.alphan.alphan import unet, alphan
from deployments.autoscaler.auto_scaler import auto_scaler
from deployments.api_infra.infra import results

deployment_config = {
    "environment_name": "environment_B",
    "deployments": [
        {
            "class": alphan,
            "name": "alphan",
            "num_replicas_base": 1,
            "num_cpus": 0,
            "num_gpus": 0,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 10,
        },
        {
            "class": unet,
            "name": "unet",
            "num_replicas_base": 2,
            "num_cpus": 1,
            "num_gpus": 0,
            "max_replicas": 4,
            "target_num_ongoing_requests_per_replica": 10,
        },
        {
            "class": auto_scaler,
            "name": "auto_scaler",
            "num_replicas_base": 1,
            "num_cpus": 0,
            "num_gpus": 0,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 10,
        },
        {
            "class": results,
            "name": "results",
            "num_replicas_base": 1,
            "num_cpus": 0,
            "num_gpus": 0,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 10,
        },
    ],
}
