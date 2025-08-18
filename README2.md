# How to dump GEM

run gdb --args ./matmul_api_demo
break rknn mem destroy
r

when break, run ./hello to dump gem to a file
try a different input format like int8 to fp16 in ./matal_api_demo
diff the difference in dump gem

find out which gem represent instruction, which for input A and inputB and outputC

attach result below

# Result

Gem 123456








# Explain


NPU uses DRM_IOCTL_GEM_FLINK to get the GEM object.

```
int fd = open("/dev/dri/card1", O_RDWR);

```



