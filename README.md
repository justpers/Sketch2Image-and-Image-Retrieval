# Sketch2image-and-Image-Retrieval
2024 D&X:W Conference CV Team
 이 프로젝트는 스케치를 통한 이미지 검색 기능을 구현한 AI 모델입니다. Sketch2Image를 통해 사용자가 갤러리에 저장된 이미지를 검색할 수 있도록 하였으며, Mediapipe, BLIP, Diffusion, DINOv2등의 기술을 사용하였습니다.

## 프로젝트 개요 & 프로젝트 Flow
- **목적**: 마우스와 키보드를 사용하지 않고 손가락 제스처로 스케치를 그려 갤러리 내 유사한 이미지를 검색할 수 있는 시스템 개발
- **주요 기능**:
    - **스케치 캡셔닝**: 손가락 제스처로 그린 스케치를 설명하는 캡션 생성
    - **디퓨전 모델**: 스케치와 스케치 캡션을 입력으로 받아 고해상도 이미지를 생성
    - **이미지 검색**: 생성된 이미지를 기반으로 갤러리에서 유사한 이미지를 검색

## 주요 모델 및 기술
**1. Mediapipe**:
   - 손가락 제스처를 추적하여 스케치 생성
   - 한 손가락으로 스케치 저장, 여러 손가락 제스처로 색상 선택 및 지우개 기능 수행
   - 상세:
     - 한 손가락 -> 스케치 저장
     - 두 손가락 -> 검정색 펜
     - 세 손가락 -> 초록색 펜
     - 네 손가락 -> 빨간색 펜
     - 다섯 손가락 -> 지우개
     - R키 -> 전체 삭제

**2. BLIP (Bootstrapped Language-Image Pre-training)**:
   - 멀티모달 모델로, 스케치에 대한 캡션 생성
   - 생성된 캡션을 디퓨전 모델의 프롬프트로 사용

**3. Img2Img-Turbo**:
   - 생성된 캡션을 바탕으로 고해상도 컬러 이미지를 생성
   - 30만 개의 예술 이미지를 사용하여 Adversarial Learning으로 학습된 모델

**4. DINOv2 (Distillation of Knowledge with Neural Operators)**:
   - ViT-H/16을 사용하여 비슷한 이미지 간의 유사도를 측정
   - query 이미지와 가장 유사한 이미지들을 검색하여 갤러리 내에서 가까운 이미지들을 검색
  
 ## 프로젝트 환경
- 환경(로컬, GPU 등등)
- 셋팅방법 각각

## DataSet
- **Sketchy Dataset**:
총 125개의 카테고리 중 39개 카테고리 선택
각 스케치에 대한 텍스트 프롬프트를 추가하여 모델 학습에 활용

## CheckPoints

- [다운로드](https://drive.google.com/drive/folders/16tHzOjyHXhN-VVOLXzvbGTfwh5uv1Sff?usp=sharing)
  
## 실행 방법

1. **환경 설정**:
   - 프로젝트의 요구 사항을 설치합니다.
     ```bash
     pip install -r requirements.txt
     ```

2. **ADB를 통해 갤럭시 장치와 연결**:
   - 안드로이드 장치에서 이미지를 `database` 폴더로 복사합니다.
     ```bash
     python main.py  # ADB를 통해 이미지를 로컬에 저장하고 Sketch2Image 시스템 실행
     ```

3. **스케치 입력 및 이미지 검색**:
   - 손가락 제스처로 스케치를 그리고, 이를 통해 캡션 생성 및 이미지 검색을 진행합니다.

## Reference
- [DINOv2 Repository](https://github.com/vra/dinov2-retrieval)
- [Img2Img-Turbo Repository](https://github.com/GaParmar/img2img-turbo)
- [Sketchy Dataset](https://github.com/CDOTAD/SketchyDatabase)
