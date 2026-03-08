This fork addresses several issues unique to ARM-based Blackwell machines:

- Flash / Sage Attention: in future will probably provide a wheel for flash which is a small win, sage for not-sm121 is not optimal 
- nunchaku: Don't, use nvfp4 (additonal nvfp4 mixed tunes may be supplied in future)
- decord: Missing aarch64 wheels in upstream. Replaced by decord2==3.0.0 which provides a compatible module interface.
- onnxruntime: onnxruntime-gpu has build issues on this platform, workaround is using the CPU version.
- taichi: No pre-built wheels, doesn't really speak gb10 so dubious value. Guarded in pyproject.toml; where it's used should fall back to CylinderRendererCPU.
- insightface: No pre-built wheels for aarch64 Linux; excluded from the Spark environment.
- torch: Very helpfully warns that 12.1 isn't 12.0

## Architecture Notes

- Performance: DGX Spark isn't great for inference, nvfp4 helps throughput but it's an aggressive quant.
- Optimization: Torch support currently treats sm121 as sm120. Flash and sage tend to underwhelm.
- Unified Memory: The mmgp package
