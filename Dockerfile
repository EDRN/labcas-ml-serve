# LabCAS ML Serve
# ===============
#
# Build with: `docker image build --tag labcas-ml-serve .`


# Base Image
# ----------
#
# Normally I'd use an Alpine image here, but Ray won't install on Alpine; see this issue for
# the sordid details: https://github.com/ray-project/ray/issues/30416
#
# In the future, if we need to support multiple Python versions, simply do
#
#     docker image build --build-arg python_version=PYTHON_VERSION --tag labcas-ml-serve:VERSION-PYTHON_VERSION
#
# and replace PYTHON_VERSION with the version of Python you like, such as 3.10.5, and VERSION
# with the version of LabCAS ML Serve being built. Repeat for each Python version needed. No
# need for pyenv.

ARG python_version=3.9.15
FROM python:${python_version}-bullseye


# Application
# -----------
#
# Copy over and install the Python-based requirements

WORKDIR /usr/src/app
COPY configs/environments/environment_B/requirements.txt requirements.txt
RUN pip install --requirement requirements.txt
COPY ./ ./


# Image Morphology
# ----------------
#
# Ports, entrypoint, etc.

EXPOSE 8080/tcp
EXPOSE 6378/tcp
EXPOSE 8265/tcp
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
# HEALTHCHECK ???


# Metadata
# --------
#
# We're "good Docker citizens"

LABEL "org.label-schema.name"="LabCAS ML Serve"
LABEL "org.label-schema.description"="LabCAS Nuclei Detector featuring machine learning and powered by Ray Serve"
LABEL "org.label-schema.version"="0.0.0"



# Posterity
# ---------
#
# Everything below this line are comments from an earlier version of this `Dockerfile` preserved
# for future generations.

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