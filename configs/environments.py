environments_config={
    "environment_A":
            {'ip': '127.0.0.2',
            'port': '6378',
            'namespace': 'serve',
            'serve_port': '8080',
            'deployments': ['/usr/src/app/deployer_env_A.py'],
            'pyenv': '/root/.pyenv/versions/environment_A/bin',
            # 'deployments': ['/Users/asitangmishra/PycharmProjects/labcas-ml-serve/deployer_env_A.py'],
            # 'pyenv': '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
            'object_store_memory': '500000000',  # 500 MB
            'num_cpus': '8',
            'dashboard-port': '8265'
            }
}
