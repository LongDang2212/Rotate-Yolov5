# Rotate - Yolov5 Deepstream 
Tested on docker image: nvcr.io/nvidia/deepstream:6.0-devel 
### Build app + lib

```bash
mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-6.0 \
    -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" \
//  -DPLATFORM_TEGRA=ON ..
make
```


### Run app

./detect file://<video_dir>

### Performance
Updating...
