from deployments.alphan.alphan import unet, alphan, preprocessing, predict_actor
from deployments.api_infra.infra import results

deployment_config={
    'environment_name': 'environment_B',
    'deployments': [
    {   'class': alphan,
        'name': 'alphan',
        'num_replicas_base': 1,
        'num_cpus': 0,
        'num_gpus': 0
    },
        {
    'class': preprocessing,
        'name': 'preprocessing',
        'num_replicas_base': 1,
        'num_cpus': 1,
        'num_gpus': 0
        },
{
    'class': predict_actor,
        'name': 'predict_actor',
        'num_replicas_base': 1,
        'num_cpus': 1,
        'num_gpus': 0
        },
    {   'class': unet,
        'name': 'unet',
        'num_replicas_base': 1,
        'num_cpus': 1,
        'num_gpus': 0
    },
    {'class': results,
     'name': 'results',
     'num_replicas_base': 1,
     'num_cpus': 0,
     'num_gpus': 0,
     }
]

}