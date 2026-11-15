# HW05 - Object Counting with Streamlit

이미지를 업로드하면 HW05.ipynb에서 사용한 영상처리 흐름을 바탕으로 객체를 분리하고 개수를 계산하는 Streamlit 웹 앱입니다.

## 주요 처리 과정

1. Grayscale 변환
2. Gaussian Blur를 이용한 Noise Reduction
3. Otsu Thresholding 및 Morphological Opening
4. Distance Transform으로 전경 마커 생성
5. Watershed로 서로 맞닿은 객체 분리
6. Connected Component 결과에 면적, 원형도, 종횡비 조건 적용
7. 검출 객체 수와 객체별 픽셀 수 출력

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더의 `app.py`, `requirements.txt`, `README.md`를 GitHub 저장소에 업로드합니다.
2. https://share.streamlit.io/ 에 접속하여 GitHub 계정으로 로그인합니다.
3. **Create app**을 선택하고 저장소, 브랜치, 메인 파일 경로(`app.py`)를 지정합니다.
4. **Deploy**를 누릅니다.
5. 배포가 완료되면 생성된 `https://....streamlit.app` 주소를 과제 제출 링크로 제출합니다.

## 참고

현재 객체 선택 조건(면적 2000 pixel 이상, circularity > 0.65, aspect ratio 0.80~1.25)은 제공된 HW05.ipynb의 마지막 동전형 객체 필터링 코드를 유지한 것입니다. 촬영 거리, 해상도, 배경 및 조명에 따라 검출 결과가 달라질 수 있습니다.
