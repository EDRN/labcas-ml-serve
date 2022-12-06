LabCAS ML Service (Beta)

Running the ML Service (run the following in a terminal. Make sure you have Git and Docker installed):  

	1. git clone git@github.com:EDRN/labcas-ml-serve.git
	cd labcas-ml-serve

    2. build docker image
    docker build -t labcas-ml-serve:1 .
	
    3. start the container and get inside
    docker run -p 6378:6378 -p 8080:8080 -p 8265:8265 -v $PWD:/usr/src/app --shm-size=1gb -it labcas-ml-serve:1 bash
    
    4. start the redis (inside the container) server in daemon mode
    redis-server --daemonize yes
    
    5. run (the following code) inside the container to start ray and deploy everything
    python src/ray_start.py
    
    6. run (the following code) inside the container to stop ray 
    python src/ray_stop.py
    
    7. Use Ctrl P+Q to exit docker container without stopping it

Access the API docs:  
http://127.0.0.1:8080/alphan/docs
http://127.0.0.1:8080/results/docs

Access the Dashboard:  
http://127.0.0.1:8265/

Try it out:  
```
The Service is hosted at: https://edrn-labcas.jpl.nasa.gov/mlserve/

# Submitting an image and gettign the task_id:
curl -u <labcas_user>:<labcas_pass> -X 'POST' 'https://edrn-labcas.jpl.nasa.gov/mlserve/alphan/predict?model_name=unet_default&is_extract_regionprops=True&window=128' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'input_image=@<IMAGE_FILE_PATH>;type=image/png'


# Getting the results:
curl -u <labcas_user>:<labcas_pass> -X 'GET' 'https://edrn-labcas.jpl.nasa.gov/mlserve/results/get_results?task_id=<task_id>' -H 'accept: application/json' -o output.zip
```

Architecture:
![alt text](https://github.com/EDRN/labcas-ml-serve/blob/main/images/labcas_ml_serve.png)
