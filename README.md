LabCAS ML Service (Beta)

Running the ML Service (run the following in a terminal. Make sure you have Git and Docker installed):  

	git clone git@github.com:EDRN/labcas-ml-serve.git
	cd labcas-ml-serve
	# build docker image
    docker build -t labcas-ml-serve:1
	# start the container and get inside
    docker run -p 6378:6378 -p 8080:8080 -p 8265:8265 -v $PWD:/usr/src/app --shm-size=1gb -it labcas-ml-serve:1 bash
    # run inside the container
    python ray_start.py
    # Use Ctrl P+Q to exit docker container without stopping it

Access the API docs:  
http://127.0.0.1:8080/alphan/docs

Access the Dashboard:  
http://127.0.0.1:8265/

Try it out:  
http://127.0.0.1:8080/alphan/predict?resource_name=8_1.png

Architecture:
![alt text](https://github.com/EDRN/labcas-ml-serve/blob/main/images/labcas_ml_serve.png)
