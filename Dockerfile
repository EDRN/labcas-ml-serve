FROM ubuntu:18.04

# START: install penv and python versions. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-on-ubuntu-18-04/
RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
RUN exec "$SHELL"
RUN pyenv install 3.7.8
# install pyenv virtualenv. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-virtualenv-on-ubuntu-18-04/
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
RUN pyenv virtualenv 3.7.8 environment_A
RUN exec "$SHELL"
# END: install penv and python versions
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

## useful commands:
# docker build -t labcas-ml-serve:1
# docker run -p 6378:6378 -p 8080:8080 -p 8265:8265 -v $PWD:/usr/src/app -it labcas-ml-serve:1 bash











#RUN mkdir -p /usr/src/app
#COPY requirements.txt /usr/src/app/
#WORKDIR /usr/src/app
#RUN wget http://download.redis.io/redis-stable.tar.gz
#RUN tar xvzf redis-stable.tar.gz
#WORKDIR redis-stable
#RUN make
#RUN make install
#WORKDIR /usr/src/app
#RUN pip install --no-cache-dir -r requirements.txt
#EXPOSE 5002

# ===== uncomment the below code to run in Production-mode:
# COPY rad /usr/src/app
# CMD ["redis-server", " --daemonize", "yes"]
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5002"]
# ===== comment till here



# ========= Important Docker commands for running and debugging:

# ==== build the docker-image:

# == Debug-mode:
# >> cd rad
# >> docker build -t rad-api:1 .

# == Production-mode:
# uncomment the last section in the dockerfile and run:
# >> cd rad
# >> docker build -t rad-api:1 .


# ==== run the docker-container and the RAD REST API:

# == Debug-mode:
# >> cd rad
# >> docker run -p 5002:5002 -v $PWD:/usr/src/app -it rad-api:1 bash
# start the redis server in daemon mode
# >> redis-server --daemonize yes
# and start api manually in a bash terminal (started above) inside the container; as a result it is a non-primary process of the container and so the container does not get killed when api is restarted!:
# >> uvicorn api.main:app --host 0.0.0.0 --port 5002

# == Production-mode:
# >> docker run -p 5002:5002 rad-api:1


# ==== some useful docker commands:

# to look for stopped containers:
# >> docker ps -a

# to start a stopped container:
# >> docker start <Container_ID>

# to detach:
# >> ctrl+p+q

# to attach:
# >> docker attach <Container_ID>

# remove all containers:
# >> docker rm $(docker ps -a -q)