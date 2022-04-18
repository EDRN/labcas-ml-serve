FROM ubuntu:18.04
# EXPOSE 8265, 8080, 6378

# install penv and python versions. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-on-ubuntu-18-04/
RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
RUN exec "$SHELL"
RUN pyenv install 3.7.8

# install and create pyenv virtualenv. Ref: https://www.liquidweb.com/kb/how-to-install-pyenv-virtualenv-on-ubuntu-18-04/
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
RUN pyenv virtualenv 3.7.8 environment_A
RUN exec "$SHELL"

# install python dependencies
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

## run commands:
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