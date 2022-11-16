# encoding: utf-8

environments_config = {
    # "environment_A":
    #         {'ip': '127.0.0.2',
    #         'port': '6378',
    #         'namespace': 'serve',
    #         'serve_port': '8080',
    #         'deployments': ['/usr/src/app/configs/environments/environment_A/deployer.py'],
    #         'pyenv': '/root/.pyenv/versions/environment_A/bin',
    #         # 'deployments': ['/Users/asitangmishra/PycharmProjects/labcas-ml-serve/configs/environments/environment_A/deployer.py'],
    #         # 'pyenv': '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
    #         'object_store_memory': '500000000',  # 500 MB
    #         'num_cpus': '8',
    #         'dashboard-port': '8265'
    #         },
    "environment_B": {
        "ip": "127.0.0.2",
        "port": "6378",
        "namespace": "serve",
        "serve_port": "8080",
        "deployments": ["/usr/src/app/configs/environments/environment_B/deployer.py"],
        "pyenv": "/root/.pyenv/versions/environment_B/bin",
        "object_store_memory": "500000000",  # 500 MB
        "num_cpus": "8",
        "dashboard-port": "8265",
    }
}
