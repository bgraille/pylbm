FROM debian:jessie

USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install gfortran libopenmpi-dev openmpi-bin openmpi-common \
                       liblapack-dev libatlas-base-dev libatlas-dev mercurial

RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update --yes conda

RUN conda create --yes -n pyenv python=3.5 numpy scipy matplotlib nose && \
    source activate pyenv && \\
    pip install -r requirements.txt && \
    pip install . 

CMD [ "/bin/bash" ]