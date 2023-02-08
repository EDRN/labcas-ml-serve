import os, os.path

_base_dir = os.getenv('ML_SERVE_HOME', '/usr/src/app')
_port = os.getenv('ML_SERVE_PORT', '8080')
_ip = os.getenv('ML_SERVE_IP', '127.0.0.2')

environments_config = {
    # "environment_A":
    #         {'ip': '127.0.0.2',
    #         'port': '6378',
    #         'namespace': 'serve',
    #         'serve_port': '8081',
    #         'deployments': ['/usr/src/app/configs/environments/environment_A/deployer.py'],
    #         # 'deployments': ['/Users/asitangmishra/PycharmProjects/labcas-ml-serve/configs/environments/environment_A/deployer.py'],
    #         # 'pyenv': '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
    #         'object_store_memory': '500000000',  # 500 MB
    #         'num_cpus': '8',
    #         'dashboard-port': '8265'
    #         },
    "environment_B": {
        'ip': _ip,
        'port': '6378',
        'namespace': 'serve',
        'serve_port': _port,
        'deployments': [os.path.join(_base_dir, 'configs/environments/environment_B/deployer.py')],
        'object_store_memory': '500000000',  # 500 MB
        'num_cpus': '8',
        'dashboard-port': '8265'
    }
}
