# -*- coding:utf-8 -*-
import os
import sys
import platform
from .rknn_log import set_log_level_and_file_path
from .rknn_platform import get_host_os_platform, get_librknn_api_require_dll_dir
from .rknn_base import RKNNBase
from argparse import Namespace

already_imported = False
for path in sys.path:
    if os.path.join('rknn', 'base') in path:
        already_imported = True
if not already_imported:
    lib_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', 'base'))
    sys.path.append(lib_path)


class RKNN:
    """
    Rockchip NN Kit
    """

    # NPU Core Mask
    NPU_CORE_AUTO = 0x0
    NPU_CORE_0 = 0x1
    NPU_CORE_1 = 0x2
    NPU_CORE_2 = 0x4
    NPU_CORE_0_1 = 0x3
    NPU_CORE_0_1_2 = 0x7
    NPU_CORE_ALL = 0xffff

    def __init__(self, verbose=False, verbose_file=None):
        if type(verbose) == str:
            verbose = verbose.lower()
        cur_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
        if get_host_os_platform() == 'Windows_x64':
            require_dll_dir = get_librknn_api_require_dll_dir()
            new_path = os.environ["PATH"] + ";" + require_dll_dir
            os.environ["PATH"] = new_path
        self.verbose = verbose
        if verbose_file is not None:
            if os.path.dirname(verbose_file) != "" and not os.path.exists(os.path.dirname(verbose_file)):
                verbose_file = None
        self.rknn_log = set_log_level_and_file_path(verbose, verbose_file)
        if verbose not in [True, False, 'debug']:
            self.rknn_log.e("'verbose' must be True or False!")
        if verbose:
            if verbose_file:
                self.rknn_log.d('Save log info to: {}'.format(verbose_file))
        self.rknn_base = RKNNBase(cur_path, verbose)

    def config(self,
               mean_values=None,
               std_values=None,
               quantized_dtype='w8a8',
               quantized_algorithm='normal',
               quantized_method='channel',
               quantized_hybrid_level=0,
               target_platform=None,
               quant_img_RGB2BGR=False,
               float_dtype='float16',
               optimization_level=3,
               custom_string=None,
               remove_weight=False,
               compress_weight=False,
               inputs_yuv_fmt=None,
               single_core_mode=False,
               dynamic_input=None,
               model_pruning=False,
               op_target=None,
               quantize_weight=False,
               remove_reshape=False,
               sparse_infer=False,
               enable_flash_attention=False,
               auto_hybrid_cos_thresh=0.98,
               auto_hybrid_euc_thresh=None,
               **kwargs,
               ):
        """
        Configs
        :param mean_values: Channel mean value list, default is None, means all mean is zero.
        :param std_values: Channel std value list, default is None, means all std is one.
        :param quantized_dtype: quantize data type, currently support: w8a8, w8a16, w16a16i, w16a16i_dfp, w4a16. default is w8a8.
        :param quantized_algorithm: currently support: normal, mmse (Min Mean Square Error), kl_divergence, gdq. default is normal.
        :param quantized_method: quantize method, currently support: layer, channel, group{SIZE}. default is channel.
                                 {SIZE} is the multiple value of 32 between 32 and 256, e.g group32.
        :param target_platform: target chip platform, currently support rv1103 / rv1103b / rv1106 / rv1106b / rv1126b / rk2118 / rk3562 / rk3566 /
                                rk3568 / rk3576 / rk3588. default is None.
        :param quant_img_RGB2BGR: whether to do RGB2BGR when load quantize image (jpg/jpeg/png/bmp), default is False.
        :param float_dtype: non quantize data type, currently support: float16, default is float16.
        :param optimization_level: set optimization level. default is 3, means use all default optimization options.
        :param custom_string: add custom string information to rknn model, then can query the information at runtime. default is None.
        :param remove_weight: generate a slave rknn model which removes conv2d weight, need share weight with rknn model of complete weights.
                              default is False.
        :param compress_weight: compress the weights of the model, which can reduce the size of rknn model. default is False.
        :param inputs_yuv_fmt: add yuv preprocess at the top of model. default is None.
        :param single_core_mode: single_core_mode=True can reduce the size of rknn model, only for rk3588. default is False.
        :param model_pruning: pruning the model to reduce the model size, default is False.
        :param op_target: used to specify the target of each operation, the format is {'111':'cpu', '222':'cpu', ...}, default is None.
        :param dynamic_input: simulate the function of dynamic input according to multiple sets of input shapes specified by the user,
                              the format is [[[1,3,224,224],[1,1,224,224], ...], [[1,3,160,160],[1,1,160,160], ...], ...].
                              default is None, experimental.
        :param quantize_weight: When 'do_quantization' of rknn.build is False, reduce the size of the rknn model by quantizing some weights.
                                default is False, is about to be deprecated!
        :param remove_reshape: Remove possible 'Reshape' in model inputs and outputs to improve model runtime performance. default is False.
        :param sparse_infer: Sparse inference on already sparsified models to improve performance. default is False.
        :param enable_flash_attention: Whether to enable Flash Attention. default is False.
        :param auto_hybrid_cos_thresh: The thresholds of cosine distance in auto hybrid when model is quantizate. default is 0.98
        :param auto_hybrid_euc_thresh: The thresholds of euclidean distance in auto hybrid when model is quantizate. default is None
        :return: success: 0, failure: -1
        """
        # Workspace args
        args = Namespace()
        import inspect
        frame = inspect.currentframe()
        args_key, _, _, _ = inspect.getargvalues(frame)
        for key in args_key:
            if key == 'self':
                continue
            setattr(args, key, locals()[key])
        args.kwargs = kwargs

        return self.rknn_base.config(args)

    def load_tensorflow(self, tf_pb, inputs, input_size_list, outputs, input_is_nchw=False):
        """
        Load TensorFlow Model
        :param tf_pb: TensorFlow model file path.
        :param inputs: Input node list, such as ['input1', 'input2'].
        :param input_size_list: Input size list, such as [[224, 224, 3], [128, 128, 1]].
        :param outputs: Output node list, such as ['output1', 'output2'].
        :param input_is_nchw: Whether the input layout of the model is already NCHW. default is False.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_tensorflow(tf_pb, inputs, input_size_list, outputs, input_is_nchw=input_is_nchw)

    def load_caffe(self, model, blobs=None, input_name=None):
        """
        Load Caffe Model
        :param model: Caffe model file path.
        :param blobs: Caffe blobs file path.
        :param input_name: Input node list, such as ['input1', 'input2'], used to specify inputs order. default is None.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_caffe(model, blobs, input_name)

    def load_tflite(self, model, input_is_nchw=False):
        """
        Load TensorFlow Lite Model
        :param model: TensorFlow Lite model file path.
        :param input_is_nchw: Whether the input layout of the model is already NCHW. default is False.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_tflite(model, input_is_nchw=input_is_nchw)

    def load_onnx(self, model, inputs=None, input_size_list=None, input_initial_val=None, outputs=None):
        """
        Load ONNX Model
        :param model: ONNX model file path.
        :param inputs: Specified model inputs, Such as: ['data']. default is None, means get from model.
        :param input_size_list: Set each input tensor size list, Such as: [[1,224,224],[3,224,224]], If inputs set, the
                                input_size_list should be set also. defualt is None.
        :param input_initial_val: Set each input initial value (ndarray list). default is None.
        :param outputs: Specified model outputs, Such as: ['resnetv24_dense0_fwd']. defualt is None, means get from model.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_onnx(model, inputs, input_size_list, input_initial_val, outputs)

    def load_darknet(self, model, weight):
        """
        Load Darknet Model
        :param model: darknet model cfg file path.
        :param weight: darknet weight file path.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_darknet(model, weight)

    def load_pytorch(self, model, input_size_list):
        """
        Load pytorch model
        :param model: pytorch traced model file path.
        :param input_size_list: Set each input tensor size list. Such as: [[1,224,224],[3,224,224]].
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_pytorch(model=model, input_size_list=input_size_list)

    def build(self, do_quantization=True, dataset=None, rknn_batch_size=None, auto_hybrid=False):
        """
        Build RKNN model
        :param do_quantization: Whether to quantize the model. default is True.
        :param dataset: DataSet file for quantization. default is None.
        :param rknn_batch_size: Batch size to inference. default is None.
        :param auto_hybrid: Whether to enable automatic hybrid quantization to adjust accuracy or overflow. default is False.
        :return: success: 0, failure: -1
        """

        return self.rknn_base.build(do_quantization=do_quantization, dataset=dataset, expand_batch_size=rknn_batch_size, auto_hybrid=auto_hybrid)

    def hybrid_quantization_step1(self, dataset='dataset.txt', rknn_batch_size=None, proposal=False, proposal_dataset_size=1, custom_hybrid=None):
        """
        Generate hybrid quantization config
        :param dataset: DataSet for quantization
        :param rknn_batch_size: batch size to inference.
        :param proposal: Generate hybrid quantization config suggestions
        :param proposal_dataset_size: The size of dataset used for proposal
        :param custom_hybrid: Generate hybrid quantization config files, the format is [[input_name, output_name], ...], default is None
        :return: success: 0, failure: -1
        """
        return self.rknn_base.hybrid_quantization_step1(
            dataset=dataset, expand_batch_size=rknn_batch_size,
            proposal=proposal, proposal_dataset_size=proposal_dataset_size, custom_hybrid=custom_hybrid)

    def hybrid_quantization_step2(self, model_input, data_input, model_quantization_cfg):
        """
        Generate rknn model info
        :param model_input: Model file path.
        :param data_input: Data file path.
        :param model_quantization_cfg: config file path for model quantization.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)

    def export_rknn(self, export_path, **kwargs):
        """
        Export rknn model to file
        :param export_path: Export rknn model file path.
        :return: success: 0, failure: -1
        """

        if 'cpp_gen_cfg' in kwargs:
            self.rknn_log.w("'cpp_gen_cfg' param of 'export_rknn' has been deprecated. Please call 'gen_cpp_demo' interface to generate cpp demo.")

        return self.rknn_base.export_rknn(export_path, **kwargs)

    def export_encrypted_rknn_model(self, input_model, output_model=None, crypt_level=1):
        """
        Encrypt a rknn model.
        :param input_model: Path of rknn model which need encrypt.
        :param output_model: Encrypted rknn model path. default is None.
        :param crypt_level: Crypt level, 1~3. default is 1. The lower the level, the faster the decryption and
                            the lower the security. The higher the level, the slower the decryption and the higher the
                            security.
        :return: success: 0, failure: -1
        """
        if input_model is None:
            self.rknn_log.e('The input_model is None! Please specify input model path!')
            return -1

        if output_model is None:
            output_model_path = os.path.splitext(input_model)[0] + '.crypt.rknn'
            self.rknn_log.w('The output_model is None, using {} as output model name.'.format(output_model_path))
        else:
            output_model_path = output_model

        if crypt_level < 1 or crypt_level > 3:
            crypt_level = 1
            self.rknn_log.w('Unsupport crypt_level {}. It should be 1, 2 or 3. Reset to default: 1.'.format(
                crypt_level))

        return self.rknn_base.get_encrypted_rknn_model(input_model, output_model_path, crypt_level)

    def load_rknn(self, path):
        """
        Load RKNN model
        :param path: RKNN model file path.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.load_rknn(path)

    def accuracy_analysis(self, inputs, output_dir='./snapshot', target=None, device_id=None):
        """
        generate inference snapshots (fp32 && qnt) and calculate quantize error.
        :param inputs: The path list of image (jpg/png/bmp/npy).
        :param output_dir: Output directory. defualt is './snapshot'.
        :param target: Currently support rv1103 / rv1103b / rv1106 / rv1106b / rv1126b / rk3562 / rk3566 / rk3568 / rk3576 / rk3588.
                       If target is set, the output of each layer of NPU will be obtained, and analyze it's accuracy.
                       default is None.
        :param device_id: if multi devices are connected, device id need be specified. default is None.
        """
        return self.rknn_base.accuracy_analysis(inputs, output_dir, target, device_id)

    def init_runtime(self, target=None, device_id=None, perf_debug=False, eval_mem=False,
                     async_mode=False, core_mask=NPU_CORE_AUTO, fallback_prior_device="cpu"):
        """
        Init run time environment. Needed by called before inference or eval performance.
        :param target: target platform, simulator or rv1103 / rv1103b / rv1106 / rv1106b / rv1126b / rk3562 / rk3566 / rk3568 / rk3576 / rk3588.
                       default is None, means simulator.
        :param device_id: adb device id, only needed when multiple devices connected to pc. default is None.
        :param perf_debug: enable or disable debugging performance, it will affect performance. defualt is False.
        :param eval_mem: enable or disable debugging memory usage, it will affect performance. default is False.
        :param async_mode: enable or disable async mode. default is False.
        :param core_mask: set npu core mask, currently support:
                          RKNN.NPU_CORE_AUTO / RKNN.NPU_CORE_0 / RKNN.NPU_CORE_1 / RKNN.NPU_CORE_2 / RKNN.NPU_CORE_0_1 / RKNN.NPU_CORE_0_1_2 / RKNN.NPU_CORE_ALL
                          default is RKNN.NPU_CORE_AUTO, only valid for rk3588, rk3576.
        :param fallback_prior_device: set fallback prior device when OP is not supported by NPU.
                                      currently support: 'gpu' or 'cpu', 'gpu' is only valid for platform which has gpu hardware.
        :return: success: 0, failure: -1
        """

        return self.rknn_base.init_runtime(target=target, device_id=device_id,
                                           perf_debug=perf_debug, eval_mem=eval_mem, async_mode=async_mode, core_mask=core_mask, 
                                           fallback_prior_device=fallback_prior_device)

    def inference(self, inputs, data_format=None, inputs_pass_through=None, get_frame_id=False):
        """
        Run model inference
        :param inputs: Input ndarray List.
        :param data_format: Data format list, current support: 'nhwc', 'nchw', default is 'nhwc', only valid for 4-dims input. default is None.
        :param inputs_pass_through: The pass_through flag (0 or 1: 0 means False, 1 means True) list. default is None.
        :param get_frame_id: Whether need to get output/input frame id in async mode, it is only available in camera demo. default is False.
        :return: Output ndarray list
        """
        return self.rknn_base.inference(inputs=inputs, data_format=data_format,
                                        inputs_pass_through=inputs_pass_through, get_frame_id=get_frame_id)

    def eval_perf(self, is_print=True, fix_freq=True):
        """
        Evaluate model performance
        :param is_print: Format print perf result. default is True.
        :param fix_freq: Whether to fix hardware frequency. default is True.
        :return: Performance Result (dict)
        """
        return self.rknn_base.eval_perf(is_print, fix_freq)

    def eval_memory(self, is_print=True):
        """
        Evaluate model memory usage
        :param is_print: Format print memory usage. default is True.
        :return: memory_detail (Dict)
        """
        return self.rknn_base.eval_memory(is_print)

    def list_devices(self):
        """
        print all adb devices and devices use ntb.
        :return: adb_devices, list; ntb_devices, list. example:
                 adb_devices = ['0123456789ABCDEF']
                 ntb_devices = ['TB-RK1808S000000009']
        """
        adb_devices, ntb_devices = self.rknn_base.list_devices()
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0:
            self.rknn_log.p('all device(s) with adb mode:')
            self.rknn_log.p(",".join(adb_devices))
        if len(ntb_devices) > 0:
            self.rknn_log.p('all device(s) with ntb mode:')
            self.rknn_log.p(",".join(ntb_devices))
        if len(adb_devices) == 0 and len(ntb_devices) == 0:
            self.rknn_log.p('None devices connected.')
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0 and len(ntb_devices) > 0:
            all_adb_devices_are_ntb_also = True
            for device in adb_devices:
                if device not in ntb_devices:
                    all_adb_devices_are_ntb_also = False
            if not all_adb_devices_are_ntb_also:
                self.rknn_log.w('Cannot use both device with adb mode and device with ntb mode.')
        return adb_devices, ntb_devices

    def get_sdk_version(self):
        """
        Get SDK version
        :return: sdk_version
        """
        return self.rknn_base.get_sdk_version()

    def reg_custom_op(self, custom_op):
        """
        Register custom operator, only supported for ONNX model.
        :param custom_op: needs to be a class similar to the following cases:
            import numpy as np
            from rknn.api.custom_op import get_node_attr
            class cstSoftmax:
                op_type = 'cstSoftmax'
                def shape_infer(self, node, in_shapes, in_dtypes):
                    out_shapes = in_shapes.copy()
                    out_dtypes = in_dtypes.copy()
                    return out_shapes, out_dtypes
                def compute(self, node, inputs):
                    x = inputs[0]
                    axis = get_node_attr(node, 'axis')
                    x_max = np.max(x, axis=axis, keepdims=True)
                    tmp = np.exp(x - x_max)
                    s = np.sum(tmp, axis=axis, keepdims=True)
                    outputs = [tmp / s]
                    return outputs
          The applicable situations:
            User needs to customize a new OP, which is not within the ONNX OP specification.
            The op_type of this new OP is recommended to start with 'cst', and the 'shape_infer'
            and 'compute' functions need to be implemented.

        :return: success: 0, failure: -1
        """
        return self.rknn_base.reg_custom_op(custom_op)

    def optimize_onnx(self, model, optimized_path=None, passes_enable=None, inputs_for_dump=None):
        """
        Optimize the onnx graph for NPU
        :param model: ONNX model file
        :param optimized_path: the path of optimized ONNX model
        :param passes_enable: the optimzie passes list that need to be enabled, default is None, means enable all passes.
        :param inputs_for_dump: input path (npy) list for dump function, if set, will dump all tensors of simulator.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.optimize_onnx(model, optimized_path, passes_enable, inputs_for_dump)

    def codegen(self, output_path, inputs=None, overwrite=False):
        """
        Generate C++ deployment example
        :param output_path: Output directory
        :param inputs: The path list of image (jpg/png/bmp/npy). Cpp demo will run with inputs if inputs is given. Default is None.
        :param overwrite: Whether to overwrite the existing file. default is False.
        :return: success: 0, failure: -1
        """
        return self.rknn_base.codegen(output_path, inputs, overwrite)

    def release(self):
        """
        Release RKNN resource
        :return: None
        """
        if platform.machine().lower() != 'amd64':
            # windows can not unsetenv
            os.unsetenv('TMPDIR')
        self.rknn_base.release()
