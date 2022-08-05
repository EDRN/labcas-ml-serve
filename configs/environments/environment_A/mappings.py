from deployments.alphan_old.alphan import remove_spur, remove_bg, remove_bg_gauss, bgnet, unet, alphan
from deployments.autoscaler.auto_scaler import auto_scaler

deployment_config={
    'environment_name': 'environment_A',
    'deployments': [
    {   'class': alphan,
        'name': 'alphan',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {   'class': remove_spur,
        'name': 'remove_spur',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {   'class': remove_bg,
        'name': 'remove_bg',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {   'class': remove_bg_gauss,
        'name': 'remove_bg_gauss',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {   'class': bgnet,
        'name': 'bgnet',
        'num_replicas_base': 2,
        'num_cpus': 1,
        'num_gpus': 0,
        'max_replicas': 4,
        'target_num_ongoing_requests_per_replica': 10
    },
    {   'class': unet,
        'name': 'unet',
        'num_replicas_base': 2,
        'num_cpus': 1,
        'num_gpus': 0,
        'max_replicas': 4,
        'target_num_ongoing_requests_per_replica': 10
    },
        {'class': auto_scaler,
        'name': 'auto_scaler',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10,
        }]

}