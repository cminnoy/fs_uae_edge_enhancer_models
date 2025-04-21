Overview: Pico and Nano Upscaling Models for FS-UAE

The pico and nano models were specifically designed and trained for use with the FS-UAE Amiga emulator.
These models aim to recreate the look and feel of original CRT displays by enhancing Amiga graphics in real time using neural upscaling.

FS-UAE Framebuffer and Display Characteristics:

FS-UAE uses a fixed framebuffer size of 752×576 pixels.
It maps all OCS, most ECS, and AGA screen modes into this framebuffer. Some modes, like SuperHiRes, are not supported by FS-UAE and are thus out of scope for these models.

The Amiga's graphics were deeply tied to the behavior of CRT displays, where the phosphorescent layer naturally blended and smoothed out coarse pixel structures.
This CRT-induced blending gave the illusion of richer color depth and resolution than the system could technically output.
Game artists often exploited this characteristic, placing individual pixels with precision to achieve visually stunning results on CRTs.

Modern LCD displays lack this smoothing effect, often resulting in harsh, blocky visuals when rendering classic Amiga graphics.
Traditional upscaling methods—such as xBRZ or other pixel-art heuristics—struggle to reproduce the nuanced appearance of CRT output.

Internal Resolution Handling in FS-UAE:

Depending on the screen mode, FS-UAE maps Amiga pixels to framebuffer pixels as follows:
    Low-resolution mode: 1 graphic pixel = 2×2 framebuffer pixels
    High-resolution mode: 1 graphic pixel = 1×2 framebuffer pixels
    Low-resolution interlaced mode: 1 graphic pixel = 2×1 framebuffer pixels
    High-resolution interlaced mode: 1 graphic pixel = 1×1 framebuffer pixel

The AI-based upscaling should be performed after the framebuffer stage BUT before the shader upscaling.
The output of the model, within FS-UAE, should be copied back into the framebuffer for further processing.

Model Details

Both models take input as a 3-channel RGB image of size 752×576.
The goal is to enhance the image in real time while preserving the artistic intent behind the original graphics.

Pico Model

The pico model prioritizes minimal latency and compact design while maintaining edge clarity.
Its architecture includes:
    Multi-scale convolutional edge detection using parallel learnable kernels of sizes 3×3, 5×5, and 7×7.
    A learnable Sobel filter to capture structural features with minimal computation.
    Two final convolutional layers focused on image enhancement and sharpening.

Performance:
    Model Size: ~56k parameters
    Runtime (AMD Radeon RX 6900 XT): 89.1 FPS

Nano Model

The nano model builds directly on pico, adding a ResNet block for deeper image correction and better handling of subtle gradients and artifacts.
It has better colour fidelity.
While the improvement is more noticeable when zoomed in or in still frames, the difference is minor during typical gameplay.

Performance:
    Model Size: ~351k parameters
    Runtime (AMD Radeon RX 6900 XT): 49.9 FPS

Final Notes

Both models succeed in enhancing the image. Use it you like it.
The choice between pico and nano depends on your needs:
    Choose pico for maximum FPS and lightweight performance.
    Choose nano if you value subtle quality improvements and your hardware can handle the load.

As always, perception is subjective, and what works best is up to the eye of the beholder.
