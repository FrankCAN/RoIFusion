# evaluation 
TFPATH=$1 # e.g: /usr/local/lib/python3.5/dist-packages/tensorflow
CUDAPATH=$2 # e.g: /usr/local/cuda-9.0
# TFPATH = /home/chencan/anaconda3/envs/roifusion/lib/python3.5/site-packages/tensorflow
# CUDAPATH = /usr/local/cuda-9.0
OPSPATH="lib/utils/tf_ops"

# evaluation
cd ${OPSPATH}/evaluation
/usr/bin/gcc-5 -std=c++11 tf_evaluate.cpp -o tf_evaluate_so.so -shared -fPIC -I ${TFPATH}/include -I ${CUDAPATH}/include -I ${TFPATH}/include/external/nsync/public -lcudart -L ${CUDAPATH}/lib64/ -L ${TFPATH} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..


