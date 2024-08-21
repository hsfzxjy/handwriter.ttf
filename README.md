<h1 align=center>handwriter.ttf</h1>
<p align=center><b> :writing_hand: A Handwriting Synthesizer by abusing <a href="https://github.com/harfbuzz/harfbuzz/blob/main/docs/wasm-shaper.md">Harfbuzz WASM Shaper</a>.</b></p>
<p align=center><b> :link: Check more stupid stuff at <a href="https://github.com/hsfzxjy/Harfbuzz-WASM-Fantasy">Harfbuzz-WASM-Fantasy</a>.</b></p>

## Introduction

During the hype of [llama.ttf](https://github.com/fuglede/llama.ttf) months ago, I was speculating the potential of WASM shaper for even crazier purpose, one that fitter to a font shaper's duty -- to synthesize font at runtime. This project as proof-of-concept implements a synthesizer that generates and rasterizes handwriting-style font, backed by [a super-lightweight RNN model](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit/blob/main/my-app/synthesis_network_52.onnx) (~14MiB).

**Usage** You may try out this project with the following steps:
1. On a Linux system with X11 (WSL is fine), run `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/hsfzxjy/handwriter.ttf`;
2. In directory `handwriter.ttf`, run `make run`;
3. Start typing in the pop-up gedit window. Each line should prefixed by `#` to trigger the shaper, e.g., typing `#hello world`.

Some strokes might look cursed due to the limitation of the model, appending a space ` ` should make it better.

https://github.com/user-attachments/assets/ea5e1b61-fcb3-4950-8621-af2499c96493

## Technical Details

### Algorithm

The project follows Alex Graves's paper [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) and adopts an RNN model for handwriting synthesis. Shortly, the generation process undergoes multiple steps to produce a series of strokes given the input text. At each step the model predicts the next pen position given the current one. Afterwards, [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) rasterizes the strokes into pixel locations, which are set as the offsets for an array of "black-box" glyphs.

I've tried some more recent models, but their runtime latency is unaffordable.

### Performance

The final TTF file is highly optimized, reaching the speed of 0.08 sec/character on Intel Ultra 125H. Each text run's generation time is proportional to the text length.

The journey to perfect optimization is interesting, which I shall introduce in blog posts later. Some important notes:

- Use [rten](https://github.com/robertknight/rten) as inference backend to make sure neural ops are executed with SIMD instructions.
- Pre-transpose the RHS of MatMul to make them col-major, improving the performance by ~15%.
- To run modules containing SIMD instructions, [wasm-micro-runtime](https://github.com/bytecodealliance/wasm-micro-runtime) should be compiled with `-DWAMR_BUILD_SIMD=1` and WASM file must be AOT-compiled by [wamrc](https://github.com/bytecodealliance/wasm-micro-runtime/tree/main/wamr-compiler).
- Enable specific optimization in `wamrc` (`--opt-level=3`, `--enable-segue=i32.load,f32.load,i32.store,f32.store` and `--enable-tail-call`), improving the performance by ~55%.

## License

This project is licensed under the [Apache 2.0 LICENSE](./LICENSE). Copyright (c) 2024 hsfzxjy.
