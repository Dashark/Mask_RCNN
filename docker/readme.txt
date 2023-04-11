# 预装nvidia-docker2

# 终端切换至Dockerfile所在目录
$ cd /home/dell/workSpace/docker-maskrcnn
$ sudo docker build -t tensorflow:maskrcnn .

# 创建并启动容器
$ sudo docker run -e PYTHONIOENCODING=utf-8 -itd --network=host -v $PWD/share:/root/share --restart=always --gpus all --tmpfs /tmpfs --name tensorflow_maskrcnn tensorflow:maskrcnn

# 进入运行的容器
$ sudo docker exec -it tensorflow_maskrcnn /bin/bash
# cd /root/share/maskrcnn_twisted/
