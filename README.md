# Question

AM device is using custom driver in tinygrad, does it need renderer?
-> yes, seems custom driver on running hip code

Given .rknn is propertary format, how ggml handles the ops and offload to npu
-> seems only matmul

how piper handles the ops and offload to npu
-> convert decoder to .rknn

how usefultransformer handles the ops and offload to npu
-> only mat mul api

reverse engineering found conv / max /min/ avg
->



# Resource

https://gist.githubusercontent.com/marty1885/a939e3cda146e333195bf8f62fde7a95/raw/a1449425d9fb03f6a97609915f4423516269484d/rknn_matmul_bench.cpp

# LICENSE
LICENSE info is currently missed. Please refer to the files and the following repo. 
A large portion of code is borrowed from

https://gitlab.freedesktop.org/tomeu/mesa/-/tree/rocket
https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/29698/diffs#27f0de65c1652925136ff56423b59469decadd21
https://github.com/phhusson/rknpu-reverse-engineering
https://github.com/mtx512/rk3588-npu
https://github.com/airockchip/rknn-toolkit2/tree/master
https://github.com/airockchip/rknn-llm/tree/main/rknpu-driver