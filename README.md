LabCAS ML Service (Beta)

Running the ML Service (run the following in a terminal. Make sure you have Git and Docker installed):  

	git clone git@github.com:EDRN/labcas-ml-serve.git
	cd labcas-ml-serve
	# 1. build docker image
    docker build -t labcas-ml-serve:1 .
	# 2. start the container and get inside
    docker run -p 6378:6378 -p 8080:8080 -p 8265:8265 -v $PWD:/usr/src/app --shm-size=1gb -it labcas-ml-serve:1 bash
    # 3. start the redis (inside the container) server in daemon mode
    # redis-server --daemonize yes
    # 4. run (the following code) inside the container to start ray and deploy everything
    python src/ray_start.py
    # 5. run (the following code) inside the container to stop ray 
    python src/ray_stop.py
    # 6. Use Ctrl P+Q to exit docker container without stopping it

Access the API docs:  
http://127.0.0.1:8080/alphan/docs
http://127.0.0.1:8080/results/docs

Access the Dashboard:  
http://127.0.0.1:8265/

Try it out:  
http://127.0.0.1:8080/alphan/predict?resource_name=8_1_0_0.png

Architecture:
![alt text](https://github.com/EDRN/labcas-ml-serve/blob/main/images/labcas_ml_serve.png)
