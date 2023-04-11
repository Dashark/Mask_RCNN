FROM tensorflow/tensorflow:1.14.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update
# opencv-python依赖 & opencv
RUN apt-get install -y libgl1 libopencv-dev
# 安装ifconfig命令
RUN apt-get install -y net-tools
# 安装ping命令
RUN apt-get install -y iputils-ping
# python env
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /root/
COPY envs ./envs
COPY Mask_RCNN ./Mask_RCNN
WORKDIR /root/envs
RUN pip3 install -r requirements.txt
WORKDIR /root/Mask_RCNN
ENTRYPOINT ["python3", "main.py"]
