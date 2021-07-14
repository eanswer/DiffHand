# FROM nvidia/cudagl:11.3.0-devel-ubuntu18.04
FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# setup timezone
RUN echo 'America/New_York' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Replacing shell with bash for later docker build commands
RUN mv /bin/sh /bin/sh-old && \
  ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         xvfb \
         tmux \
         libgtk2.0-dev \
         ca-certificates \
         libjpeg-dev \
         freeglut3-dev  \
         wget \
         software-properties-common \
         libopenblas-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


COPY ./install_python_3_7.sh /tmp/install_python_3_7.sh
RUN chmod +x /tmp/install_python_3_7.sh && /tmp/install_python_3_7.sh
COPY ./install_diffhand.sh /tmp/install_diffhand.sh
RUN chmod +x /tmp/install_diffhand.sh && /tmp/install_diffhand.sh
COPY ./install_py_packages.sh /tmp/install_py_packages.sh
RUN chmod +x /tmp/install_py_packages.sh && /tmp/install_py_packages.sh



RUN rm -rf /tmp/*
WORKDIR /workspace

CMD ["bash"]