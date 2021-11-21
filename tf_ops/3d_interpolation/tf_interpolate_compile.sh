# TF1.4 Python 3.5 CUDA 8.0
# TF1.4 Python 3.7 CUDA 11.1

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_interpolate.cpp \
-o tf_interpolate_so.so \
-shared \
-fPIC \
-I$TF_INC \
-I/usr/local/cuda-11.1/include \
-I$TF_INC/external/nsync/public \
-lcudart \
-L/usr/local/cuda-11.1/lib64/ \
-L$TF_LIB \
-ltensorflow_framework

#g++ -std=c++11 tf_interpolate.cpp \
#-o tf_interpolate_so.so \
#-shared \
#-fPIC \
#-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include \
#-I/usr/local/cuda-8.0/include \
#-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include/external/nsync/public \
#-lcudart \
#-L/usr/local/cuda-8.0/lib64/ \
#-L$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow \
#-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
