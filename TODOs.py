"""
1. Currently, the Dockerfile contains hardcoded python versions and environment names, this can be read from
 configs/environments/basic.py and should be automated to support multiple environments; a for loop in the Dockerfile
  I guess :)
2. There is a deployer.py in every environment dir eg. configs/environments/environment_A/deployer.py, this is used by
 the create_environments function, where it calls this file via commandline. This is quite roundabout and  redundant.
    should ideally be done programmatically!
3. Documentation needs to be created on how to include a new environment and deployment into this framework.
 BTW, currently, multiple deployments can be mapped to a single environment! There is also a way where Ray supports
  multiple venvs, but for the same python version! this is useful when different deployments need different versions
   of a library! This needs to be used and made explicit here.
4. Code needs to be added for auto-scaling down the resources here: deployments/autoscaler/src/auto_scaler.py. The code
    for auto-scaling up is already there.
    (Note:- It is possible Ray now has this functionality inbuilt. We have implemented this as a hack, using a
    deployment for monitoring and scaling using information on requests volume!)
5. the get_logger function is within the deployment, create a utility file from where this should be accessed,
 then channel the logs to a particular output directory!
 6. Get rid of all the sys.path.insert statements, look also for os.path.abspath(__file__)
"""