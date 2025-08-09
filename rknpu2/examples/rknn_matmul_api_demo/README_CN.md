rknn_matmul_api_demo是一个使用matmul C API在NPU上执行矩阵乘法的示例。包含静态shape和动态shape两种模式。

用法:

1. 静态shape模式

   ```
   Usage:
   ./rknn_matmul_api_demo <matmul_type> <M,K,N> <B_layout> <AC_layout> <loop_count> <core_mask> <print_result> <iommu_domain_id>
        matmul_type = 1: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
        matmul_type = 2: RKNN_INT8_MM_INT8_TO_INT32
        matmul_type = 4: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16
        matmul_type = 7: RKNN_FLOAT16_MM_INT4_TO_FLOAT32
        matmul_type = 10: RKNN_INT4_MM_INT4_TO_INT16
   Example: A = [4,64], B = [64,32], int8 matmul test command as followed:
   ./rknn_matmul_api_demo 2 4,64,32
   ```
2. 动态shape模式

   ```sh
   ./rknn_matmul_api_dynshape_demo <matmul_type> <M1K1N1#M2K2N2#...> <B_layout> <AC_layout> <loop_count> <core_mask>
        M_shapes:         M shape array, which separeted by ',' 
        matmul_type = 1: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
        matmul_type = 2: RKNN_INT8_MM_INT8_TO_INT32
        matmul_type = 4: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16
        matmul_type = 7: RKNN_FLOAT16_MM_INT4_TO_FLOAT32
        matmul_type = 10: RKNN_INT4_MM_INT4_TO_INT16
   Example: A = [1,64]#[4,64]#[8,64], B = [64,32], int8 matmul test command as followed:
    feature+const: ./rknn_matmul_api_dynshape_demo 2 1,64,32#4,64,32#8,64,32 1 1
    two feature:   ./rknn_matmul_api_dynshape_demo 2 1,64,32#4,64,32#8,64,32 2 1
   ```

# Aarch64 Linux 示例

## 编译

首先导入GCC_COMPILER，例如`export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu 
`，然后执行如下命令：

```
./build-linux.sh -t <target> -a <arch> -b <build_type>]

# 例如: 
./build-linux.sh -t rk3588 -a aarch64 -b Release
```
## 安装

将 install/rknn_matmul_api_demo_Linux 拷贝到设备上。

- 如果使用Rockchip的EVB板，可以使用以下命令：

连接设备并将程序传输到`/userdata`

```
adb push install/rknn_matmul_api_demo_Linux /userdata/
```

- 如果你的板子有sshd服务，可以使用scp命令或者其他方式将程序和模型传输到板子上。

## 运行


```
adb shell
cd /userdata/rknn_matmul_api_demo_Linux/
```

```
export LD_LIBRARY_PATH=./lib
./rknn_matmul_api_demo 2 4 64 32
```

或者

```sh
export LD_LIBRARY_PATH=./lib
./rknn_matmul_api_dynshape_demo 2 1,64,32#4,64,32#8,64,32 1 1
```



# Android 示例

## 编译

首先导入ANDROID_NDK_PATH，例如`export ANDROID_NDK_PATH=~/opts/ndk/android-ndk-r18b`，然后执行如下命令：

```
./build-android.sh -t <target> -a <arch> [-b <build_type>]

# 例如: 
./build-android.sh -t rk3568 -a arm64-v8a -b Release
```

## 安装

连接设备并将程序传输到`/data`

```
adb push install/rknn_matmul_api_demo_Android /data/
```

## 运行

```
adb shell
cd /data/rknn_matmul_api_demo_Android/
```

```
export LD_LIBRARY_PATH=./lib
./rknn_matmul_api_demo 2 4 64 32
```
或者
```sh
export LD_LIBRARY_PATH=./lib
./rknn_matmul_api_dynshape_demo 2 1,64,32#4,64,32#8,64,32 1 1
```

