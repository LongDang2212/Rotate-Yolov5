# Rotate - Yolov5 TensorRT 

### Build app + lib

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
// -DPLATFORM_TEGRA=ON ..
make -j$(nproc)
```

### Export TRT

    ./export ../../weights/rotate_best.onnx ryolo.engine

### Inference
    ./infer ryolo.engine <image_dir>
    
### Latency and performance 
Updating...

