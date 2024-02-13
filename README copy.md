# Introduction

이 프로젝트는 CenterNet를 경량화 하기 위해 만들어졌습니다.
1. Backbone을 통째로 교체하는 방법
2. Hourglass의 resisual block을 fire module로 교체하는 방법
3. Head의 conv 연산을 depth-wise separable conv로 교체하는 방법

위 세 가지 방법을 시도하였습니다.

# Installation

Docker를 이용하여 환경을 설정합니다.

Step 1.

    git clone https://github.com/samkwon1122/CenterNet-light.git

Step 2.

    cd CenterNet-light
    docker build -t mmdetection docker/
    docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data -v {WORK_DIR}:/mmdetection/work_dirs mmdetection

데이터는 다음과 같이 구성해줍니다. 본 프로젝트에서는 Pascal VOC 데이터를 사용하였습니다.

    data
    ├── VOCdevkit
    │   ├── VOC2007
    │   │   ├── Annotations
    │   │   ├── ImageSets
    │   │   ├── JPEGImages
    │   ├── VOC2012
    │   │   │   ├── Annotations
    │   │   ├── ImageSets
    │   │   ├── JPEGImages

# Training

/mmdetection/configs/centernet-lite 폴더에 모델 별 config 파일이 있습니다.

다음과 같은 방법으로 학습을 진행합니다.

    python tools/train.py ${CONFIG_FILE}

예시

    python tools.train.py configs/centernet-lite/centernet_hgsq104_light.py

학습이 진행되면 /mmdetection/work_dirs/ 경로에 log 및 checkpoint 파일이 저장됩니다.

# Test

학습이 완료된 모델의 성능을 테스트 합니다.
    
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

예시

    python tools/test.py configs/centernet-lite/centernet_hgsq104_light.py work_dirs/centernet_hgsq104_light/epoch_10.pth

# Demo

특정 이미지에 대한 추론 결과를 확인할 수 있습니다.

    python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} --weights ${WEIGHTS}

예시

    python demo/image_demo.py demo/demo.jpg configs/centernet-lite/centernet_hgsq104_light.py --weights work_dirs/centernet_hgsq104_light/epoch_10.pth

# 학습 조건
- GPU: NVIDIA GeForce RTX 3060
- Epochs: 50
- Learing Rate: 2.5e-4, 40epoch 이후로는 2.5e-5
- Optimizer: Adam
- Batch Size: 16 (Hourglass-104의 경우 메모리 부족으로 6)
- 모든 모델 동일하게 학습 진행

# Results




