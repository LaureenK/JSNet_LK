# TF1.4 Python 3.5 CUDA 8.0
# TF1.4 Python 3.7 CUDA 11.1

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#/bin/bash
/usr/local/cuda-11.1/bin/nvcc tf_grouping_g.cu \
-o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp \
tf_grouping_g.cu.o \
-o tf_grouping_so.so \
-shared -fPIC \
-I$TF_INC \
-I/usr/local/cuda-11.1/include \
-I$TF_INC/external/nsync/public \
-lcudart -L/usr/local/cuda-11.1/lib64/ \
-L$TF_LIB \
-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
