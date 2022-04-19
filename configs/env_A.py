from alphan import remove_spur, remove_bg, remove_bg_gauss, bgnet, unet, alphan
from auto_scaler import auto_scaler

class_name_mappings={
'alphan': alphan,
'remove_spur': remove_spur,
'remove_bg':remove_bg,
'remove_bg_gauss':remove_bg_gauss,
'bgnet':bgnet,
'unet':unet,
'auto_scaler':auto_scaler
}

deployment_config={
    'environment_name': 'environment_A',
    'deployments': [
    {   'name': 'alphan',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {
        'name': 'remove_spur',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {
        'name': 'remove_bg',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {
        'name': 'remove_bg_gauss',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10
    },
    {
        'name': 'bgnet',
        'num_replicas_base': 2,
        'num_cpus': 1,
        'num_gpus': 0,
        'max_replicas': 4,
        'target_num_ongoing_requests_per_replica': 10
    },
    {
        'name': 'unet',
        'num_replicas_base': 2,
        'num_cpus': 1,
        'num_gpus': 0,
        'max_replicas': 4,
        'target_num_ongoing_requests_per_replica': 10
    },
        {
        'name': 'auto_scaler',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0,
        'max_replicas': 1,
        'target_num_ongoing_requests_per_replica': 10,
        }]

}