https://clehaxze.tw/gemlog/2024/03-01-optimizing-socs-for-llm-on-the-edge.gmi

Martin's blog/website thingy

Home
Blog/Gemlog
Public Services/Demo
whoami
Old blog
Github
Gemini
Optimizing SoCs for Large Language Models on the edge
Around December, 2023. I had a quick talk with a chip maker about porting and what hardware features is needed for fast LLM inferencing - due to my work on porting llama.cpp to the RK3588 NPU. I started writing this post as the condensation of my view and recommendations. But.. I got busy and forgot completely. Until Prof.Chang, my advisor during college asked me to be a guest lecturer in his course on the same topic(s). So here we are. I'll share the slides when that happens.

This post assumes you are familiar with the internals of deep learning and have knowledge of computer architectures like cache, DMA and how they are interconnected. If you don't, this post will make little sense. Lots of HPC jargons will be used to convey to convey precise meaning. By no means what I share is SOTA. But simply what I see during my development. Maybe they are simple, plain facts to you.

DISCLAIMER: This piece of work have nothing to do with my employer. I am writing this on my own time and because I don't want a 300W heat source in my room during summer. Opinions are my own, etc.. etc.. etc.. you know the drill.
The anatomy of LLaMA and RWKV
As of early 2024. There are 2 classes of large language models. Transformers and what I call "massive feed forward". The former is more popular because that's where the LLM boom begins, GPT-2/3/4, LLaMA, ChatGLM are all based on Transformers. While the lattar are emerging and tries to solve the massive compute and memory needed by Transformers. Examples includes RWKV and Mamba. Despite architectural differences, these models are similar in the sense that, like older vision models, complex blocks are stacked on top of each other to achieve better and better accuracy. However, unlike vision models where convolution is main operation, LLMs run on massive matrix multiplications. Popular 7 billion parameter models have individual weights of up to 4096 x 10240. On top of that, transformers generates 2 matrices called the key and query matrix at runtime then multiplied together to produce the result. Prohibiting certain hardware optimizations.

Let's look into how a single layer of LLaMA is constructed. The following code is taken from llama-from-scratch as it is the most readable version of LLaMA I can find. I've annotated it with regards to how computation is applied.

# Taken from https://github.com/bkitano/llama-from-scratch/blob/main/llama.ipynb
# Codeblock 18. 

config = {
    'batch_size': 10,
    'd_model': 512,
    'n_heads': 8,
    'context_window': 16,
}

# This is the core of the model
class RoPEAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Each "layer" in LLaMA is 3 large matrix multiplications. the Q, K and V matrix
        # In our example, each a 512 x 512 matrix. In actual LLaMA, this is 4096 x 4096
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        # The rotary matrix is a 3D matrix of size 16 x 512 x 512. In this example we only have
        # context_window = 16. In practice it has to be something like 512 or larget to be anywhere
        # near useful.
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        # 5 matrix multiplications performed on input
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        # The key attention step. This step is super expensive as it generates
        # a large N x N matrix where N is the context window.
        # This is where projects like FlastAttention and CTransformer tries to 
        # optimize. In GGLM, this is still implemented as a matrix multiplication
        # as they find it to not be the bottleneck for them.
        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1
        )
        # Naive scaled_dot_product_attention is approximately as heavy as:
        # (q_rotated @ k_rotated.transpose(-2, -1)) @ v 

        if return_attn_weights:
            # Same here, attention weights is N x N
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

layer = RoPEAttentionHead(config)
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)
And the following is the Attention as RWKV-v5 implements it. Code is taken and restructured from tinyrwkv's v5 implementation. The FFN block works on roughly the same computations so I won't include it here.

class Att:
    def __init__(self, n_heads, head_dim, time_mix_k, time_mix_v,
        time_mix_r, key, value, receptance, time_first, time_decay,
        gn_weight, gn_bias, output):
        ...

    def __call__(self, x, att_xx, att_ss) -> tuple[Tensor, Tensor, Tensor]:
        # Lite vector addition and multiplication
        xk = self.time_mix_k * (x - att_xx) + att_xx
        xv = self.time_mix_v * (x - att_xx) + att_xx
        xr = self.time_mix_r * (x - att_xx) + att_xx

        # 3 matrix multiplications. Reshape is computationally free
        k = (self.key @ xk).reshape(self.n_heads, self.head_dim, 1)
        v = (self.value @ xv).reshape(self.n_heads, 1, self.head_dim)
        r = (self.receptance @ xr).reshape(self.n_heads, 1, self.head_dim)

        ss = att_ss.reshape(self.n_heads, self.head_dim, self.head_dim)

        # Mixing, another set of matrix multiplications
        a = k @ v
        o = r @ (self.time_first * a + ss)
        o = self.output_group_norm(o.flatten().unsqueeze(0)).squeeze(0)

        return (
            # A final matrix multiplication
            self.output @ o,
            x,
            (a + self.time_decay * ss).flatten(),
        )
class Block:
    def __init__(self, ...):
        ...
    
    def __call__(self, x, att_xx, att_ss, ffn_xx) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Note the 2 LayerNorms here. This is another part of heavy computation in RWKV
        ln1 = self.att_ln(x) # Here
        att, att_xx, att_ss = self.att(ln1, att_xx, att_ss)
        x = x + att
        ln2 = self.ffn_ln(x) # And here
        ffn, ffn_xx = self.ffn(ln2, ffn_xx)
        x = x + ffn

        return (x, att_xx.realize(), att_ss.realize(), ffn_xx.realize())
Nomatter the underlying architecture, these blocks are stacked on top of each other to form the final model like how it works in in VGG or ResNet, learning capability is gained just by stacking more and more layers. However, unlike vision models, LLMs almost solely rely on matrix multiplications - one of the main reason why convolutional neural networks are invented in the first place is because of the massive compute and memory needed to run a fully connected network. But this time there is no trick around it. Convolution doesn't work for language processing. We must run matrix multiplications. And this is where the problem lies.

In addition to matrix multiplication. Transformers needs dot product attention which is hard to optimize. Unlike in a feedforward layer, where one of the matrix is fixed and thus can be pre-reordered to optimize cache access, in attention, both matrices are generated at runtime. Limiting the amount of optimizations back to classic BLAS. RWKV although don't have dot product attention, it has the problem of having a pile of LaterNorms on very long vectors. Though, this problem is much easier to solve than dot product attention.

How GGML optimizes for bandwidth
Traditionally in vision models, since the input image is always a LDR (Low Dynamic Range) image between 0~255. The weights tend to fall in the range of -1~1. This nicely fits into a int8, whithout loosing much accuracy. However, this approach won't fly with LLMs for 2 reasons. One, the accuracy loss has significant impact on the model's performance. Two, traditional processors only does math between the same type. When the weight is int8, either the input has to be mapped to int8 or the weight has to be mapped to FP32. In the first case, LLMs are too sensitive to input precision. In the second case, you are wasting SIMD width and completely nullify the benefit of int8 if bulk conversion is used. So, GGML uses a different approach, but the naming is confusing at best.

Several quantization methods are supported. They differ in the resulting model disk size and inference speed.
<followed by a table of model performance in F16, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0>
| Model | Measure      | F16    | Q4_0   | Q4_1   | Q5_0   | Q5_1   | Q8_0   |
|------:|--------------|-------:|-------:|-------:|-------:|-------:|-------:|
|    7B | perplexity   | 5.9066 | 6.1565 | 6.0912 | 5.9862 | 5.9481 | 5.9070 |
|    7B | file size    |  13.0G |   3.5G |   3.9G |   4.3G |   4.7G |   6.7G |
|    7B | ms/tok @ 4th |    127 |     55 |     54 |     76 |     83 |     72 |
|    7B | ms/tok @ 8th |    122 |     43 |     45 |     52 |     56 |     67 |
|    7B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
|   13B | perplexity   | 5.2543 | 5.3860 | 5.3608 | 5.2856 | 5.2706 | 5.2548 |
|   13B | file size    |  25.0G |   6.8G |   7.6G |   8.3G |   9.1G |    13G |
|   13B | ms/tok @ 4th |      - |    103 |    105 |    148 |    160 |    131 |
|   13B | ms/tok @ 8th |      - |     73 |     82 |     98 |    105 |    128 |
|   13B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
F16 is actually 16bit floating point. But you'll be wrong if you assume Q8_0, Q5_0 Q4_0 is INT8, INT8 and INT4. It's fine, I assumed that before. And got educated by weird numbers that kept popping up and segfaults. Also, how come the bits per wright is not an integer?? The details of how quantization works can be found in the following link. The author had kindly provided a reference implementation and I've annotated it to the best of my ability.

llama.cpp #1240 - QX_4 quantization
#define QK4_4 128              // The QK4_4 quantization type quantizes 128 weights at a time
typedef struct {
    int8_t  scales[QK4_4/8];   // quantized scales per 8 weights   
    uint8_t qs[QK4_4/2];       // nibbles / quants of the "super-block"       
    ggml_fp16_t d;             //  "super-block" scale  
} block_q4_4;

static void dequantize_row_q4_4(const void * restrict vx, float * restrict y, int k) {
    // k is the number of weights pointed to by `vx`. Which must be a multiple of QK4_4
    assert(k % QK4_4 == 0);
    // nb stands for number of blocks
    const int nb = k / QK4_4;
          
    const block_q4_4 * restrict x = vx;
    
    uint32_t u;
    // For each block
    for (int i = 0; i < nb; i++) {
        // read the scale of the super block
        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * q = x[i].qs;
    
        // For each block in the super block
        for (int n = 0; n < QK4_4/8; ++n) {
            // Load q, which is then decomposed into 8, 4 bit numbers
            // memcpy is used to avoid alignment issues
            memcpy(&u, q, 4);
            const uint32_t u1 = (u >> 0) & 0x0f0f0f0f;
            const uint32_t u2 = (u >> 4) & 0x0f0f0f0f;

            // Now we point into the decomposed q
            const int8_t * v1 = (const int8_t*)&u1;
            const int8_t * v2 = (const int8_t*)&u2;
            // Calculate the scale of the block by multiplying the super block scale with the scale of the block
            float d = d_all * x[i].scales[n];
            // Apply the scale to the weights
            y[0] = d * (v1[0] - 8);
            y[1] = d * (v2[0] - 8);
            y[2] = d * (v1[1] - 8);
            y[3] = d * (v2[1] - 8);
            y[4] = d * (v1[2] - 8);
            y[5] = d * (v2[2] - 8);
            y[6] = d * (v1[3] - 8);
            y[7] = d * (v2[3] - 8);

            // advance the pointers
            q += 4;
            y += 8;
        } 
    }
}
That's still too complicated to my liking. Even the reference implementation is optimized. Lemme write my own implementation in C++. Assuming we are using the same block_q4_4 structure.

// grabs the n-th nibble from u
// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
// |           32 bit int          |
int8_t get_nibble(uint32_t u, int n)
{
    return (u >> (n * 4)) & 0x0f;
}

static void dequantize_single_q4_4(const block_q4_4& block, std::span<float> out)
{
    assert(out.size() == 128);

    const float d_all = GGML_FP16_TO_FP32(block.d);
    const uint32_t* q = block.qs; // UB, but whatever

    // For each block in the super block
    for (int n = 0; n < 16; ++n) {
        uint32_t u = q[n];
        int8_t scale = block.scales[n];
        float d = d_all * scale;

        // Each weight is 4 bits in `u`. We extract the 4 bits and apply the scale
        // Minus 8 is to center the weights around 0, the same idea as IEEE754's mantissa
        for (int i = 0; i < 8; ++i) {
            int8_t nibble = get_nibble(u, i);
            out[n * 8 + i] = d * (nibble - 8);
        }

    }
}

static void dequantize_row_q4_4(const block_q4_4* blocks, std::span<float> out)
{
    assert(out.size() % 128 == 0);
    for (int i = 0; i < out.size(); i += 128) {
        dequantize_single_q4_4(blocks[i / 128], out.subspan(i, 128));
    }
}
So that's quantization. How does GGML accelerate matrix multiplications? No CPU or GPU supports this crazy nested quantization format. Let's look at the relevant function.

ggml_vec_dot_t    const vec_dot               = type_traits[type].vec_dot;
...
for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
    for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
        for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
            for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                vec_dot(ne00, &tmp[ir0 - iir0], src0_row + ir0*nb01, src1_col);
            }
        }
    }
}

void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        uint8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) a[l+ 0] = q4[l] & 0xF;
        for (int l = 0; l < 32; ++l) a[l+32] = q4[l]  >> 4;

        const uint16_t * restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * GGML_FP16_TO_FP32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d[0]);

        for (int j = 0; j < QK_K/32; ++j) {
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            q8 += 16; a += 16;
            for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
            q8 += 16; a += 16;
            const float dl = d * scales[j];
            for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l+8]);
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}
Let's ignore the nonsense naming. We can see the same quantization code as above. But now it seems to be doing some multiply-accumulate with sums[l] += dl * (aux16[l] + aux16[l+8]). And that's exactly right. Older versoins of GGML actually decompresses the quantized weights back into FP32 and do chunked dot product on it. The code I shown, from a more recent commit, forgoes the step and directly does dot product on the quantized weights. And what's repeated dot prodcut? Matrix multiplcation.

This can't be good for the instruction cache nor be using SIMD efficiently. And yes, you are right. Although GGML does provide hand written SIMD code for quantized dot product, it still feels suboptimal as a lot of cycles are "wasted" on decompressing the weights.

That's until you remember we are bottlenecked by memory bandwidth. Turns out doing all the decompression in software is actually faster than reading more data from memory. Furthermore, since the decompression is applied chunk by chunk, realistically the decompressed results won't be written back to memory, ideally they will be kept in L1D$ or L2$. Thus the memory system never observes the decompressed weights. This is one of the most cleaver tricks I've seen in a while.

My 2 cents on designing a LLM accelerator
There's 2 main class of optimizations depending on your processor's processing speed and power budget. Since compute-in-memory is still far away, can't beat raw memory bandwidth, any LLM inference accelerators have to optimize for what it can.

Generally, there are a few traits that will be beneficial to LLMs and can be added to current accelerators:

As high memory bandwidth as possible
If it has a cache, consider adding some sort of streaming support so weights won't occupy cache space
Or just use a scratchpad and deal with the complexity in software
Consider low precision math (FP8) to reduce power of MACs
Optimize for GEMV throughput, GEMM is not as important (though things are happening with efficient batched inference)
Some hyper specific optimizations that can be done, but may not be a good use of area if you are aiming for a general purpose accelerator:

Hardware support for weight decompression
Support for mixed precision math. Ex: out = FP16 * FP8 + FP16
Or better, direct mixed FP16/FP8 multiply-accumulate against quantized weight
More, wider buses and memory lanes
And depending on if the processor can saturate memory bandwith, you will want to optimize for different things:

Saturated → Focus on for reduced power consumption and fast weight decompression
Not saturated → Optimize for memory access and GEMV throughput
In the case where memory is saturated. You want to put the accelerator as close to the CPU as possible. This way the CPU can deal with weight decompression, store the wrights back to L2$ and not use up memory. Or you just let the accelerator deal with the weight decompression.

However, in both cases, low precision math reduces both area and power. It's a good idea to have support for it. Yet that's it's own can of worms. 8 bit is not enough accuracy for the entire inference. Usually a pre-processing step is needed to determine which layer needs more mantissa bits and sacrifice the fractions. It's unlike FP16 where you can just throw everything at it and expect it to just work.

There's one pesky issue with RWKV that it doesn't work well with GGML quantization. It's not a problem with the quantization itself, but the fact that RWKV is not a transformer and what works for LLaMA isn't working that well for RWKV. But I think that'll be fixed in no time.

Be prepard for radical change
A preprint just came out claiming 1.58bits per wright (ternery) is enough for LLMs. If true and is applicable to later generations of LLaMA models. We might unlock a brain new scaling path for LLMs. With 1.58 bits per weight, we can fit LLaMA2-7B into ~1.5 GB of memory. With GDDR6 bandwidth, we are looking at a theatrical 400 tokens/s. Though the proposed method requries training from scratch. But it's a small price to pay for massive speed improvements and power savings (no more multiply-accumulate, only accumulate).

arXiv: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
Where the RK3588 NPU stands
Going back to my project of porting llama.cpp to the RK3588 NPU. Now it should be apparent why it does not boast well. First, their NPU uses DMA to move data in and out of the NPU. But will hit the same memory bandwidth limit quite soon. Secondly, the NPU only supports INT8 and FP16. FP16 is too much bits per weight and INT8 is not accurate enough. Finally, issue of the NPU itself, the matrix multiplication on it is horrible. A mere 10 GFLOPS for FP16 and 20 TOPS for INT8. Even the CPU can beat that.

All in all, the RK3588 NPU is not a good fit for LLM inference. I hope Rockchip can sort of mitigate the issue in software and make it somewhat usable. But you know why it doesn't work well now.

Advice for hardware vendors
Simple. MAKE YOUR SDK OPEN SOURCE AND SELL YOUR HARDWARE TO DEVELOPES WHO WANTS ONE

I don't care how great you think your new, shiny chip is. Nor how good your business plan is. Not even how good your team.. Give open access to your SDK and let the tremendous community do your work for you. I cannot stress this enough. The AI/ML/HPC community is full of smart people who can do wonders with hardware. The major problem we are facing is the lack of accessibility to the hardware. Next being no open source SDK that be (ab)used for literal magic. Just look at what I did for RK3588. I would've been 10x easier if the SDK is open source. Or the opsn source VIP9000 driver. They had to reverse engineer the entire thing before the community can do anything with it. Making the NPU a literal dead weight.

Some vendors are even wose then this. It's pain in the a** to even get a working hardware. And when you do, they won't give you the SDK without a contract. Guess how NVIDIA won the GPGPU race? They made CUDA super accessible. AMD to this day is still playing catch up. Intel being far behind. Even NVIDIA has to rely on the community to make their hardware shine. What makes you think you can solo it? - you can't.

Or, to quote myself:

Paid developer time is extremely expensive.
FOSS developers has much more time on hand. The only issue is you can't decide what the FOSS developer will do. But that shouldn't matter since it's always something added to your product.

Cool ideas in the space and outlook
I've been looking through the literature and found some cool ideas I think will be beneficial for inference.

New numerical formats that gives more precision when the weight is NOT close to 0
Sub-byte floating point formats to furthered reduce power
Reduced power on the control plane. Get rid of the cache since dataflow is predictable
Efficient memory systems. Apple M series is doing very well in this regard
Finally, I want to mention a conflict between current use cases and hardware optimizations. Some applications, epically like code assistants, interactive code generators, need to be fast and responsive. Amortizing costs across batched inference is not an option. While this kind of optimization can be quite useful for question answering and chatbots. This is like with vision models. Some applications need to be fast and responsive. Choose you battles wisely.

(Self promotion) Call for action/donations
I have worked on support for the RK3588 NPU in llama.cpp. I'm currently blocked by Rockchip's SDK issues. And is looking to pivot my work to other platforms. If you are a vendor with some intresting hardware. And is willing to give me access to your SDK. I'm more than happy to port llama.cpp to it. IMO this is a mutually beneficial relationship. GGML is popular among the broader LLM community. There's already mature support to convert and interface with LLMs running on top of GGML. GGML is also generic, in the sense that getting one model accelerated means many other LLM also gets accelerated. For me, I considering doing FOSS a donation (see me quote above, developer time is expensive) and I believe optimizing edge LLMs is a way to get power consumption of AI down quick. Which will help to slow down climate change. And the resulting spead of LLMs will help researchers and developers to do more with less.

If you are interested, you can find my contact information on the following page.

The "whoami" page on my blog
If you are viewing this post over Gemini, use this link instead
That's it. I hope my post is useful to someone on the Internet. It'll be quite interesting to see how people is going to cite a random blog post. Epically if it's over Gemini. Will publishers even accept it? Hehehe... evil laugh. I'll see you when I have something else to write about. Now get out of my house!

Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)