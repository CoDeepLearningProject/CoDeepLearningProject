import cv2
import numpy as np
import scipy.fftpack
import timeit

##2021/05/10
##이미지 처리 방식 -> Homomorphic filter
##이미지 처리 -> 영상 처리
##현재 이슈: 이미지 처리 프레임당 0.4초씩 걸림 -> 안정화 필요

def preprocessing(img):
    # 사이즈 크롭 - 눈 비율과 머리 각도 영상 사이즈와 동기화하기
    #img = img[:, 59:cols-20]

    # 이미지 row, col 개수 불러오기 (위에서 크롭한 기준이다)
    rows = img.shape[0]
    cols = img.shape[1]

    # 로그 연산으로 바꿔주기 -> 곱 연산을 덧셈 연산으로 바꾸기 위함이다
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # 시그마 값을 10으로 줘서 가우시안 분자식을 만듬
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    # Low pass와 High pass 필터 생성
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    #fftshift의 역연산
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # 필터 적용, 크롭
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = np.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

    # 감마값을 파라미터로, 이 감마값을 조정함으로써 adaptive한 이미지를 적용할 수 있다
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

    # 처음에 로그값을 적용하여 곱셈 -> 덧셈 연산으로 만들어 주었는데, exp를 통해 다시 원래대로 돌려주면 된다
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    return Ihmf2

if __name__ == "__main__":
    cap = cv2.VideoCapture("./SampleVideo.mp4")

    if cap.isOpened():                 # 캡쳐 객체 초기화 확인
        while True:
            ret, img = cap.read()      # 다음 프레임 읽기
            if ret:                     # 프레임 읽기 정상
                start_t = timeit.default_timer()

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame = preprocessing(frame)

                terminate_t = timeit.default_timer()
                timePer = ((terminate_t - start_t))

                cv2.imshow('video_file', frame) # 화면에 표시
                print('이번 프레임: ' + str(timePer))
                cv2.waitKey(1)            # 파라미터와 영상 속도는 반비례
            else:                       # 다음 프레임 읽을 수 없슴,
                break                   # 재생 완료
    else:
        print("can't open video.")      # 캡쳐 객체 초기화 실패

    cap.release()                       # 캡쳐 자원 반납
    cv2.destroyAllWindows()

