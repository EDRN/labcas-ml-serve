import os

deployments_info={

    "example_A":
            {'ip': '127.0.0.1',
            'port': '6378',
            'namespace': 'serve',
            'serve_port': '8080',
            'deployer': '/Users/asitangmishra/PycharmProjects/labcas-ml-serve/examples/envs_3_7_8/deployer.py',
            'penv': '/Users/asitangmishra/PycharmProjects/alpha_n/venv/bin',
            'object_store_memory': '500000000',  # 500 MB
            'dashboard-port': '8265'
            },

        "example_B" :
            {'ip': '127.0.0.1',
            'port': '6379',
            'namespace': 'serve',
            'serve_port': '8081',
            'deployer': '/Users/asitangmishra/PycharmProjects/labcas-ml-serve/examples/envs_3_9_0/deployer.py',
            'penv': '/Users/asitangmishra/PycharmProjects/labcas-ml-serve/venv/bin',
            'object_store_memory': '500000000',  # 500 MB
            'dashboard-port': '8266'
            }

}


def do_deployments():
    for deployment_name, deployment_info in deployments_info.items():
        os.system("source "+os.path.join(deployment_info['penv'], 'activate')+
                     " && ray start --head  --port "+deployment_info['port']+" --object-store-memory "+deployment_info['object_store_memory']+
                     " && python "+deployment_info['deployer'])


def kill_deployments():
    for deployment_name, deployment_info in deployments_info.items():
        os.system("source "+os.path.join(deployment_info['penv'], 'activate')+
                     " && ray stop")


if __name__ == "__main__":
    kill_deployments()
    do_deployments()
