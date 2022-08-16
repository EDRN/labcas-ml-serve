FROM ubuntu:18.04

# install penv and python versions.
RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
ENV HOME="/root"
WORKDIR ${HOME}
RUN apt-get install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
ENV PYTHON_VERSION=3.9.0
RUN pyenv install ${PYTHON_VERSION}

# install and create pyenv virtualenv.
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
RUN pyenv virtualenv 3.9.0 environment_B

# install python dependencies
RUN pyenv global environment_B
COPY configs/environments/environment_B/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# install redis
WORKDIR /usr/src/app
RUN wget http://download.redis.io/redis-stable.tar.gz
RUN tar xvzf redis-stable.tar.gz
WORKDIR redis-stable
RUN make
RUN make install

# start the service
WORKDIR /usr/src/app

## ==== run commands:
# docker build -t labcas-ml-serve:1
# docker run -p 6378:6378 -p 8080:8080 -p 8265:8265 -v $PWD:/usr/src/app --shm-size=1gb -it labcas-ml-serve:1 bash

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


# ==== commands for setting things up manually in linux:

##  install penv and python versions. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-on-ubuntu-18-04/
# apt update -y
# DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
# git clone https://github.com/pyenv/pyenv.git ~/.pyenv
# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
# echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
# exec "$SHELL"
# pyenv install 3.7.8

## install and create pyenv virtualenv. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-virtualenv-on-ubuntu-18-04/
# git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
# pyenv virtualenv 3.7.8 environment_A
# RUN exec "$SHELL"