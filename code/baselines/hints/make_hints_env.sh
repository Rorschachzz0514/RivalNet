#!/bin/bash
# 为 HINTS (WWW 2021, TensorFlow 1.x) 建立独立环境: python 3.7 + tensorflow 1.15.5 (CPU)
set -e
/opt/anaconda3/bin/conda create -y -p /path/to/envs/hints python=3.7 pip -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels
P=/path/to/envs/hints/bin/pip
$P install -q tensorflow==1.15.5 "numpy<1.19" "protobuf<3.21" scipy==1.4.1 pandas==1.1.5 scikit-learn==0.22.2 tqdm pyarrow==6.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
/path/to/envs/hints/bin/python -c "import tensorflow as tf, numpy, scipy, pandas, pyarrow; print(tf.__version__, numpy.__version__, scipy.__version__, pandas.__version__, pyarrow.__version__); print(tf.test.is_gpu_available())"
echo "ENV EXIT=0"
