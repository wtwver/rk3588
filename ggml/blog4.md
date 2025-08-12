Martin's blog/website thingy

Home
Blog/Gemlog
Public Services/Demo
whoami
Old blog
Github
Gemini
Accelerating Piper text-to-speech on the RK3588 NPU
Ho ho ho. Happy Hollidays! With Rockchip releasing rknn-toolkit2 1.6.0, the feature set becomes more and more complete. In this release, it's enough to be used to accelerate the Piper text to speech system. I want to document what I've done to make it work, what's my vision for it and what's more to be done.

I named the final project "Paroli" after the Esperanto word for "speak". You can find the source code here:

Paroli - Streaming TTS based on Piper with optional RK3588 NPU support
This all started out when I saw the following PR on the Piper repo, Piper being a fast, local and high quality TTS system. GitHub user mush42 wrote streaming support for Piper. This is a very important for a few reasons. First, it allows Piper to be used in a more real-time setting where the original system has to run one sentence at a time. Second, it makes the slow part, the decoder, more-or-less a static graph. Enabling old-fashioned, vision focused neural accelerators to be used. I was thinking about if I can use the RK3588's NPU to accelerate it. And use it as a part of my ongoing project to build my own, privacy focused, voice assistant.

ONNX streaming support for Piper
I started messing around with mush42's demo code. Piper works in a very interesting way. Unlike most end to end TTS systems, Piper is hybrid. It does not just take text (ASCII/UTF-8 code points) and output audio. It first passes it through espeak-ng to get phoneme sequences. Then that sequence is mapped to a per-model phoneme ID. Then the ID sequence is finally passed to the synthesizer. I first tried passing hand crafted garbage to the encoder and expecting garbage output - at least confirming streaming is working. It failed. Pure silence and 0 output from the encoder. I thought the entire code was broken until I tried passing a real sentence to it. Dang.

As a baseline, I was programming on my ex-employer's workstation - I love my ex-boss - it's an Threadripper 1800X system. Yields a real time factor of ~0.2 using the CPU. And ~0.65 on the RK3588 CPU. Not to shabby if you ask me. But that's already known as Piper will run at almost real time on a Raspberry Pi 4. Now the question is, can we do better?


Youtube video: Thorsten-Voice - Piper on Raspberry Pi 4
My first reaction is to drop RKNN (Rockchip's inference engine) in and replace the ONNX powered decoder with it. It took me quite a while to understand what mush42's code is doing. There's some mysterious moving window and audio stitching going on. Eventually I figured out 2 things. First, even though the decoder is static graph, the input shape is still somewhat dynamic, it takes up to WINDOW_SIZE + 2*OVERLAP_SIZE frames, but can be as low as 1 single frame. Second, the overlap is so that the decoder has enough context to make a good synthesis. Afterwards, the initial and final OVERLAP_SIZE frames are discarded. That leads to the first problem, RKNN needs a static input shape, specsified at model compilation time. The solution? Pad the input so it is always the maximum size. Then cut the result to the correct size. This is not ideal and with unwanted overhead, but it works.

As a side note. RKNN does have a feature called "dynamic shape" but it's a list of input shapes that the model will allow during inference, and I did try. That lead to problems after problems. Their C API throws C++ exception whenever I apply input data to a dynamic shape model. I had to GDB their Python library to figure out what's the correct API call sequence to use, all without symbols and looking at registers. And only after I got it running, learned that it's utterly broken on 1.6.0. It's a long story and I'll spare you.

Using RKNN instead of ONNX got me to a real time factor of ~0.15 on the RK3588 NPU. A whole 4.3x speedup over the CPU. And faster then the ThreadRipper 1800X! Woohoo! Yay baby, I'm golden. But sure, it's not that easy. There are popping and cracking sounds in the output. I've gone back and cheked both the CPU output and the original Piper's output. The original Piper code works flawlessly, but mush42's streaming code also them. Indicating this is a problem with streaming. Looking at the audio in Audacity shows that the popping and cracking sounds are actually side effects of the crude audio stitching.

Take the following clip as an example.

Audio: Example of cracking sound using mush42's streaming synthesis: "All the so-called “Linux” distributions are really distributions of GNU/Linux."
Audacity view of the cracking sound, it is easy to see the sudden jumps in the sample
Image: Audacity view of the cracking sound, it is easy to see the sudden jumps in the sample
I'll get back to that later. Another, more important technical issue I need to solve is that Python is slow and I need the TTS to be a REST API, or better, WebSockets so I can stream in real time. And actual multithreading to support concurrent synthesis. I took the liberty to rewrite the entire thing in C++. RKNN's C API is.. interesting. The mountain of questionable design is unreal. Partially because the RK3588 although have a 3-core NPU. They act independently and can't work together to make a single inference faster.

To use the NPU in C/C++. First, initialize a RKNN context. I'll ignore all error handling for now as it's not important.

// model is a std::vector<uint8_t> containing the model data
rknn_context ctx;
rknn_init(&ctx, model.data(), model.size(), 0, nullptr);
Then, query the model's input and output properties. This is where the pain starts. Instead of a parameter in the rknn_query function to ask for property of the i-th input or output. You have to set the index value in the rknn_tensor_attr struct and pass it to the function. Why...

rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

input_attrs.resize(io_num.n_input);
output_attrs.resize(io_num.n_output);
inputs.resize(io_num.n_input);
outputs.resize(io_num.n_output);

for(int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i; // <--- Why?
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
}

for(int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i; // <--- Like, seriously, why?
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
}
Inference is where the document lies to you. In the RKNN user guide 1.6.0. It says that you can a float array to the input buffer when the input expects FP16. No, the C API does not support that. You have to convert yourselves. That only works in Python. The stupid index write happens again.

for(int i = 0; i < input_attrs.size(); i++) {
    auto& attr = input_attrs[i];
    auto& buffer = buffers[i];

    inputs[i].index = attr.index;
    inputs[i].size = buffer.size() * sizeof(__fp16);
    inputs[i].type = RKNN_TENSOR_FLOAT16;
    inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[i].buf = buffer.data();
}

rknn_inputs_set(ctx, input_attrs.size(), inputs.data());
To get the output, you have to call rknn_outputs_get then rknn_outputs_release to free the output buffer. Like, can you just reuse the buffer? Or manage it youtselves? Or return a pointer so I can use it instead of managing the memory in a really weird way?

// RKNN expects me to know the output size beforehand, fair
rknn_outputs_get(ctx, output_attrs.size(), outputs.data(), nullptr);
...
// But I have to release the buffer myself? Sure but now I have a chunk of "free" memory
// that is owned by my vector. I either have to clear() it so I can't access it anymore
// or remember that before the next rknn_outputs_get call, I can't touch the contents.
// clear()-ing works but querying the size of output each time is unwarranted overhead.
rknn_outputs_release(ctx, output_attrs.size(), outputs.data());
And finally, since there are 3 cores and the only way they work together is when you have a batch size of 3 and each core takes 1 input. Multi-core does not help with reducing time-to-first-sample latency. And 6x realtime is fast enough most use cases. I ended up duplicating the context and let each of them handle different requests. Luckly RKNN does support sharing the same weights across contexts. So it's not that bad. However the order of API calls is important. You have to call rknn_dup_context before any query calls. Otherwise segfault.

rknn_context dup_ctx[3];
dup_ctx[0] = ctx; // Anti-RAII
for(int i = 1; i < 3; i++) {
    memset(&dup_ctx[i], 0, sizeof(dup_ctx[i]));
    rknn_dup_context(&ctx, &dup_ctx[i]);
}
With RKNN inference implemented in C++ and hooked it up to REST and WebSockets. The next problem to tackle is compression. So far Piper generates PCM samples and either directly dumps them or slaps on a WAV header then dumps them. That's not ideal considering how much bandwidth raw uncompressed audio needs. I have no prior experience with audio compressing. As far as I know there are common formats like OGG, FLAC and MP3. After much digging, turns out OGG is jsut a container format. Internally it's Vorbis or OPUS. And MP3 is not even close to either one of them in terms of quality. FLAC on the other hand only work with multi-channel audio. So I went with OPUS. Sources say it's better then Vorbis in almost every way.

I tried directly using libogg and libopus.. It works but doesn't really. The resulting audio will play in MPV. But crashes Windows Media Player. And sometimes they glitch. I can't figure out why or how to fix them. Even read the OGG specs. Then I found libopusenc. It's a wrapper around libopus and libogg making them much easier to use. However, it is a rather new library and not incuded in Ubuntu until 23.04. I had to build it from soure and add a dependency not in the package manager. But it works and I hope people will switch to Ubuntu 24.04 when that comes out.

Now back to dealing with the popping sound. My first thought was, since the popping sound is caused by audio samples being far apart from each other. I can simply look around and find a point in the newly generated audio where the sample values are similar. It helps but does not completely eliminate the issue. Looking at the audio in Audacity again, I noticed that the popping sound is caused by a sudden change in the sample's gradient. Like a triangle wave's tip. I added another term to take account of the delta between samples before and after the stitching point. Still have issues.. Many, many tries and they all failed.

That is until I rethink my approach. Instead of trying to figure out the correct stitching point. Why not look at the newly generated audio. And find the point where it's the most similar to the old audio. That way both sample values and gradients are taken into account. Sort of like running convolution on the audio. Now that works. The popping sound is gone. And I'm able to limit the search space to less then 100 samples with a much simpler heuristic. Stiching is now faster then previous multi-variable seach.

If you still hear popping. That's the model's fault not stitching.

Audio: The result of my fixed stitching algorithm: "C++ is a high-level, general-purpose programming language created by Danish computer scientist Bjarne Stroustrup."
That's most of the story. Latency is good, synthesizing speed is good and power draw is very good. With a TDP of 7W, I have no problem running in my bedroom or closet. I've also sent patches to useful-transformers to update their Whisper support for the newer RKNN library. Now I just need to hook up them to a LLM and experiement with how I can utilize them. Here's a demo I uploaded to YouTube.


Youtube video: Me on YouTube: Low latency and accelerated Text to Speech on the RK3588 NPU
Here are more samples of the synthesizer. I can't release the model as 1. It's not mine to release. 2. They are made by my friend and uses questionable training data. But hope you can hear how well it works. All of them running on the NPU and much faster then real time.

Audio: Random text 1: "Given a set of items, each with a weight and a value, determine which items to include in the collection so that the total weight is less than or equal to a given limit and the total value is as large as possible."
Audio: Random text 2: "That is the Piazza del Quirinale, in Rome, Italy. It is one of the most famous piazzas in Rome, and it is home to the Quirinale Palace, the official residence of the President of the Italian Republic."
Audio: Random text 3: "A synchronous condenser, also known as a synchronous capacitor or compensator, is a large piece of equipment used in electrical power grids to regulate voltage and stability. While it functions similarly to a conventional generator, it serves a different purpose."
Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)