# CoDeepLearningProject

## Roles
* 김찬호, 남기훈 - Dilb 활용 EAR 계산 및 메인 스레드 작성. 경보음과 평균값을 위한 서브 스레드 작성. 이미지 전처리 메서드 작성.
* 강병희, 정미서 - HeadPose 계산 코드 작성 및 FPS 저하문제 해결.
* 이진영 - MediaPipe FaceMesh를 활용하여 Dlib의 문제점 개선 및 WHENet, YOLO모델 기반 Head Post 계산 모듈 작성.

## Spec
* OpenCV 4.5.2 
* Python 3.6

## Main Algorithm
![image](https://user-images.githubusercontent.com/53031059/132081752-5821ece5-cb4f-4a77-83ae-731428a9c21e.png)

## Preprocess
* Camera frame에서 원활한 특징점 검출을 위해 조명 제거 진행
![image](https://user-images.githubusercontent.com/53031059/132082002-1f3b52e4-788f-460f-b7f1-6240c3ed4be1.png)


* LAB, RGB로 각각 전환
* LAB 변환 파일에 medianBlur 적용
* RGB와 Compose(3:1) 적용

## Calculate EAR
* Sub thread에서 EAR threshold 계산 (사용자마다 custom한 threshold를 적용하기 위함)
* Main thread에서는 threshold와 그 시점의 계산된 EAR을 비교하여 졸음 감지 진행.

### Face Mesh vs Dlib
* Dlib을 사용하였을 때 각도에 따라 인지되지 않는 경우가 있음
* 특징점 검출에 필요한 시간은 비슷함 (Dlib이 살짝 빠름)
* 더 넓은 각도에서도 정확함을 요구하는 것이 중요하다고 판단해서 Face Mesh를 사용하기로 결정

### EAR 
```
def eye_aspect_ratio(eye) :
    P1P5 = distance.euclidean(eye[1], eye[5])
    P2P4 = distance.euclidean(eye[2], eye[4])
    P0P3 = distance.euclidean(eye[0], eye[3])

    return (P1P5 + P2P4) / (2.0 * P0P3)
```

## Head Pose
* WHENet 오픈소스 참조 기존의 HeadPose 모듈과 비교했을 때 훨씬 빨라지고, 정확해진 Head Pose Detection 가능.
* BIWI 데이터셋을 사용하여, +- 75degree까지 고려.

![image](https://user-images.githubusercontent.com/53031059/132082127-b388dfda-a05b-4da2-aa6d-bfad2ab07137.png)

## TODO
* Camera의 fps와 차량의 현재속도 등을 고려하여 Time threshold를 결정하는 과정이 추가되면 좋을 듯 하다.

