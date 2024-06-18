FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
# RUN adduser --system api
# USER api
WORKDIR /gritapi
EXPOSE 8888
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*
COPY detectron2 /gritapi/detectron2/
WORKDIR /gritapi/detectron2
RUN pip install -e .
COPY GRiT/requirements.txt /gritapi/GRiT/
WORKDIR /gritapi/GRiT
RUN pip install -r requirements.txt
COPY . /gritapi/
CMD python wsgi.py