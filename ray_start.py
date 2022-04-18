import os

environments_info={

    "environment_A":
            {'ip': '127.0.0.1',
            'port': '6378',
            'namespace': 'serve',
            'serve_port': '8080',
            'deployer': '/usr/src/app/alphan.py', # '/Users/asitangmishra/PycharmProjects/labcas-ml-serve/examples/envs_3_7_8/deployer.py',
            'penv': '/root/.pyenv/versions/environment_A/bin', # '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
            'object_store_memory': '500000000',  # 500 MB
            'dashboard-port': '8265'
            }

}


def do_deployments(head=False):
    for environment_name, environment_info in environments_info.items():
        os.system( ". "+os.path.join(environment_info['penv'], 'activate')+ # Ref: https://unix.stackexchange.com/questions/246813/unable-to-use-source-command-within-python-script
                     " && ray start  --port "+environment_info['port']+" --object-store-memory "+environment_info['object_store_memory']+
                     " --head" if head else ""+
                     " && python "+environment_info['deployer'])


def kill_deployments():
    for environment_name, environment_info in environments_info.items():
        os.system( ". "+os.path.join(environment_info['penv'], 'activate')+
                     " && ray stop")


if __name__ == "__main__":
    kill_deployments()
    do_deployments(head=True)
