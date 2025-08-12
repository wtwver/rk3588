https://clehaxze.tw/gemlog/2023/10-22-experiemtal-rknpu2-backend-for-ggml-llamacpp.gmi

Martin's blog/website thingy

Home
Blog/Gemlog
Public Services/Demo
whoami
Old blog
Github
Gemini
Experimental RKNPU2 backend for GGML/llama.cpp
This weekend, after a night of partying with my friend and somehow ending up hanging out at a near by McDonald. Back to home, I picked up my old work of running LLMs on the Rockchip RK3588's NPU. Last time I hacked around directly withing GGML and running the RWKV model. That was quite a failure, slow, etc.. I have drafted a blog post about it, but never got motivated to finish. That project ended up as a tech talk at SkyMizer, and you can find the slide in the following link. Armed with the experience and a quarter bottle of Vodka in my system, I started my mad hacking to get LLaMA 2 running on the RK3588's NPU.

Porting LLM to RK3588 - Slides about my previous attempt at SkyMizer
Right, now I'm a believer of Vodka, Wiskey stinks. And here's the source code.

My development fork of llama.cpp with the RKNPU2 backend
I would say to overall process of adding a new backend is pleasant. If you are comfortable with pure C programming. I'm not old enough to experience the early days of Linux. But I'd say GGML feals like early day Linux. The structure is quite loose and things are constantly moving. But there's a certain flow to it. It's easy to find where you add your code stepping through it in your head is quite easy. If not, printf is always there. There's also barely any abstraction and documentation.

To summarize the chaotic night:

Add ggml-rknpu2.{c,h} file to the project
Add CMake options to enable the backend
Use the OpenCL backend as a template (small, simple, and easy to understand)
Find where the backends are initialized and tensor are allocated
Add init and transform functions for RKNPU2
Find where matrix multiplication is done, add new code path for RKNPU2
Figure out details about GGML
It's late night, like 3:30AM when I got the skeleton working. I'm pretty sure the amount of Vodka is more of a liability even tho it does a good job impeding me overthinking and just "fucking do it". Suprsiingly no hangover the next morning. Matrix multiplcation is running, just generating pure gibberish. I wasn't paying attention to how data is layed out in GGML and RKNPU2 yesterday anyway.

This is when I run into my a real trouble. My experience running the NPU in FP16 mode tells me it sucks. So I opted to use int8 this round. I assumed the quantization level Q8_0 is int8. But the stride makes 0 sense. Why is the stride of a 4096 x 4096 matrix to be 34, 11696??? In the past I relied on the fact that floating point NaN will show up if I read the wrong data. This is not the case for int8. AddressSanitizer can't help either since the pointer I got points into a chunk of file mapped memory. I can't really read "out of bounds".

I started reading more of the GGML source code. And I found the snippet in llama.cpp

if (ggml_is_quantized(tensor->type)) {
    qtype = ggml_internal_get_type_traits(tensor->type);
    if (qtype.to_float == NULL) {
        throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
    }
} else if (tensor->type != GGML_TYPE_F16) {
    throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
}
Wait a second.. If the only type that doesn't get dequantized is FP16. Then what is Q8_0? Tracing ggml_internal_get_type_traits leads me to the type_traits array. Stating

[GGML_TYPE_Q8_0] = {
    .type_name                = "q8_0",
    .blck_size                = QK8_0,
    .type_size                = sizeof(block_q8_0),
    .is_quantized             = true,
    .to_float                 = dequantize_row_q8_0,
    .from_float               = quantize_row_q8_0,
    .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
    .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
    .vec_dot_type             = GGML_TYPE_Q8_0,
},
So Q8_0 is actually a quantized type. I had to use ggml_internal_get_type_traits(type).to_float to dequantize the data, before I can quantize it again for the NPU. With that mystery solved, LLaMA starts to generate valid output. That is, with degraded quality and 10% less throughput.

Ok, I'm not disappointed at all. Last time I ported RWKV to the NPU, I got 2x slower speed than the CPU. 10% is great. The degraded quality is annoying though. Running 10 layers of the LLaMA2-7B model on the NPU is fine. But running all of them generates gibberish. I suspect it is caused by the quantization. RKNPU2 only supports 8bit integer, so I have to treat both the weight and input as 8 bit fixed point.

This where I'm at right now. It works, but it is not practical to use. Rockchip needs to improve the SDK capability. We need faster matrix multiplication and support for mixing floating point and int8 multiplications. I can only hope Rockchip will improve the SDK. But with their track record, I am not expect anything in the near 3 months.

LLaMA2-7B running on the RK3588's NPU. But slow and inaccurate.
Image: LLaMA2-7B running on the RK3588's NPU. But slow and inaccurate.
Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)