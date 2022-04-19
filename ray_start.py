import os

environments_info={
    "environment_A":
            {'ip': '127.0.0.1',
            'port': '6378',
            'namespace': 'serve',
            'serve_port': '8080',
            # 'deployments': ['/usr/src/app/alphan.py', '/usr/src/app/auto_scaler.py']
            # 'pyenv': '/root/.pyenv/versions/environment_A/bin',
            'deployments': ['/Users/asitangmishra/PycharmProjects/labcas-ml-serve/alphan.py', '/Users/asitangmishra/PycharmProjects/labcas-ml-serve/auto_scaler.py'],
            'pyenv': '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
            'object_store_memory': '500000000',  # 500 MB
            'num_cpus': '8',
            'dashboard-port': '8265'
            }
}


def create_environments(head=False):
    for environment_name, environment_info in environments_info.items():
        command= ". "+os.path.join(environment_info['pyenv'], 'activate')+\
                     " && ray start" +\
                 " --port "+environment_info['port']+\
                 " --object-store-memory "+environment_info['object_store_memory']+\
                 " --num-cpus "+environment_info['num_cpus']+\
                     (" --head" if head else "")+\
                    "".join([" && python "+deployment for deployment in environment_info['deployments']])

        # Ref: https://unix.stackexchange.com/questions/246813/unable-to-use-source-command-within-python-script
        print('RUNNING on shell:', command)
        os.system(command)


def kill_environments():
    for environment_name, environment_info in environments_info.items():
        command=". "+os.path.join(environment_info['pyenv'], 'activate')+\
                     " && ray stop"
        print('RUNNING on shell:', command)
        os.system(command)

if __name__ == "__main__":
    kill_environments()
    create_environments(head=True)
