# mnist-cnn-project-01
딥러닝 응용 과제 - MNIST CNN 성능 개선

# MNIST CNN 모델 성능 개선 프로젝트
'딥러닝 응용' 과제 리포트용 코드입니다.

## 1. 프로젝트 설명
residual_gap 아키텍처(97.24%)에서 시작하여 하이퍼파라미터 튜닝, 아키텍처 변경(Depthwise/Pointwise)을 거쳐  최종 99.20%의 테스트 정확도를 달성했습니다.

## 2. 최종 결과
* **Test Accuracy: 99.20%**

## 3. 코드 실행 방법
train_optimizer.py 로 학습 후 test.py 로 결과 확인

1. 필요한 라이브러리 설치
`pip install torch`
‘Pip install numpy’
