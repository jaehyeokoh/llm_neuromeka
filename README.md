
# How to install

## environment used in project
   OS : Ubuntu 22.04.5LTS\
   Python : 3.10.12\
   Camera : Intel D435i\
   cuda : 12.1

## main파일 설치법
1. main 폴더로 이동
2. uv sync 터미널에 입력
3. requirements에 있는게 자동으로 설치될 것이고 파이썬 버전 3.10.12의 venv가 활성화 됨을 확인

## mcp 서버 설치법

### 1. detector server 설치법
1. detector_server 폴더로 이동
2. 터미널에 uv venv --python 3.10.12 입력해 venv생성
3. requirements 에 명시된 pip들 설치
4. cuda 12.1 버전 설치
5. 아래 명령어 입력 후 설치
```
uv pip install torch==2.5.1+cu121torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

6. https://github.com/IDEA-Research/Grounded-SAM-2 들어가 clone 후 사이트의 안내에 따라 설치
7. https://github.com/ChaoningZhang/MobileSAM 들어가 clone 혹은 사이트 안내에 따라 설치 후 ppt에 나온대로 pt파일 배치

### 2. Anygrasp server 설치법
1. anygrasp_server 폴더로 이동
2. 터미널에 uv venv --python 3.10.12 입력해 venv생성
3. requirements 에 명시된 pip들 설치 (설치중 실패하는 경우가 있을수도 있음 e.g numpy, cv2 그럴 경우 발생 시 해당 모듈은 일단 설치에서 제외 - 어짜피 나중에 Anygrasp 설치할 때 깔림)
4. cuda 12.1 버전 설치 (만약 안했으면)
5. 아래 명령어 입력 후 설치

```
uv pip install torch==2.5.1+cu121torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```
6. Install the MinkowskiEngine.
   git clone https://github.com/NVIDIA/MinkowskiEngine.git

7. After install Minkowski Engine, Find the following files and Add headers.
```
1) **.../MinkowskiEngine/src/convolution_kernel.cuh**  
   Add header:
   #include <thrust/execution_policy.h>

2) **.../MinkowskiEngine/src/coordinate_map_gpu.cu**  
   Add headers:
   #include <thrust/unique.h> 
   #include <thrust/remove.h>

3) **.../MinkowskiEngine/src/spmm.cu**  
   Add headers:
   #include <thrust/execution_policy.h> 
   #include <thrust/reduce.h> 
   #include <thrust/sort.h>
```
8. After change, put export MAX_JOBS=2 in terminal (Minkowski 빌드할때 컴퓨터 과부화로 멈추는거 방지)
9. Run this in terminal to build MinkowskiEngine
```
cd MinkowskiEngine 
uv pip install --upgrade setuptools==59.8.0 
uv pip install ninja 
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
10. https://github.com/graspnet/anygrasp_sdk 사이트를 들어가 먼저 license 등록 후 사이트의 안내대로 설치

11. license는 답장이 오기까지 3~7일 소요됨 만약 답장이 오면 license 폴더에 전부 넣기

12. anygrasp sdk/grasp detection/gsnet_versions에 들어가 gsnet.cpython-310-x86_64-linux-gnu.so, lib_cxx.cpython-310-x86_64-linux-gnu.so를 복사해 가장 상위 폴더(anygrasp_server)에 붙혀넣기

### 3. Indy server 설치법
1. indy_server 폴더로 이동
2. uv sync 터미널에 입력
3. requirements에 있는게 자동으로 설치될 것이고 파이썬 버전 3.10.12의 venv가 활성화 됨을 확인




