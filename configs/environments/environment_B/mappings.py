from deployments.alphan.alphan import nuclei_detector_unet_128, nuclei_detector_unet_64, nuclei_detector_unet_256, alphan, predict_actor
from deployments.sam.run import sam_default, sam_predict_actor, sam
from deployments.samed.run import samed_default, samed_predict_actor, samed
from deployments.api_infra.infra import results

deployment_config={
    'environment_name': 'environment_B',
    'deployments': [
    {   'class': alphan,
        'name': 'alphan',
        'num_cpus': 0,
        'num_gpus': 0,
        'autoscaling_config': {
         "min_replicas": 1,
         "initial_replicas": 1,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
{
    'class': predict_actor,
        'name': 'predict_actor',
        'num_cpus': 1,
        'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
        },
    {'class': nuclei_detector_unet_64,
        'name': 'nuclei_detector_unet_64',
        'num_cpus': 1,
        'num_gpus': 0,
     'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
    {'class': nuclei_detector_unet_128,
        'name': 'nuclei_detector_unet_128',
        'num_cpus': 1,
        'num_gpus': 0,
        'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
    {'class': nuclei_detector_unet_256,
        'name': 'nuclei_detector_unet_256',
        'num_cpus': 1,
        'num_gpus': 0,
        'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
    {'class': results,
     'name': 'results',
     'num_cpus': 0,
     'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 1,
         "initial_replicas": 1,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
     },
{   'class': sam,
        'name': 'sam',
        'num_cpus': 0,
        'num_gpus': 0,
        'autoscaling_config': {
         "min_replicas": 1,
         "initial_replicas": 1,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
{
    'class': sam_predict_actor,
        'name': 'sam_predict_actor',
        'num_cpus': 0,
        'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
        },
    {'class': sam_default,
        'name': 'sam_default',
        'num_cpus': 1,
        'num_gpus': 0,
     'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
    },
{
    'class': samed_default,
        'name': 'samed_default',
        'num_cpus': 1,
        'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
        },
{
    'class': samed_predict_actor,
        'name': 'samed_predict_actor',
        'num_cpus': 0,
        'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 0,
         "initial_replicas": 0,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
        },
{
    'class': samed,
        'name': 'samed',
        'num_cpus': 0,
        'num_gpus': 0,
    'autoscaling_config': {
         "min_replicas": 1,
         "initial_replicas": 1,
         "max_replicas": 1,
         "target_num_ongoing_requests_per_replica": 10,
         "downscale_delay_s": 600 # seconds
     }
        }
]

}
