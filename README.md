# AI_Work Supervisor

### **Pose Estimation을 기반으로 한 작업 현장 관리 시스템 (건설현장에서의 실시간 자세 파악 및 신체 부담 측정과 위험상황 감지를 중점으로)**

### Using:

- [OpenCV](https://opencv.org/)

- [Mediapipe](https://google.github.io/mediapipe/)

- [scikit-learn](https://scikit-learn.org/stable/)

## *MediaPipe Pose*

from ... [Pose - mediapipe (google.github.io)](https://google.github.io/mediapipe/solutions/pose)

### Person/pose Detection Model (BlazePose Detector)

The detector is inspired by our own lightweight [BlazeFace](https://arxiv.org/abs/1907.05047) model, used in [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html), as a proxy for a person detector. It explicitly predicts two additional virtual keypoints that firmly describe the human body center, rotation and scale as a circle. Inspired by [Leonardo’s Vitruvian man](https://en.wikipedia.org/wiki/Vitruvian_Man), we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints.

| <img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_detector_vitruvian_man.png" title="" alt="pose_tracking_detector_vitruvian_man.png" data-align="center"> |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *Fig 1. Vitruvian man aligned via two virtual keypoints predicted by BlazePose detector in addition to the face bounding box.*                                                    |

### Pose Landmark Model (BlazePose [GHUM](https://github.com/google-research/google-research/tree/master/ghum) 3D)

The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (see figure below).

<img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png" title="" alt="pose_tracking_full_body_landmarks.png" data-align="center">
<br>
Fig 2. 33 pose landmarks.

## To Do:

- [ ] 작업 행동 분석/정형화하여 판단 모델 만들기

- [ ] 실험 및 성능평가

- [ ] 데모 시스템 만들어보기

- [ ] 결론 및 보완할 점 파악





---

참고 문헌:

- CNN기반의 학습모델을 활용한 거북목 증후군 자세 교정 시스템 [[링크]](http://koreascience.or.kr/article/JAKO202022560454953.page)
- 감시 영상을 활용한 OpenPose 기반 아동 학대 판단시스템 [[링크]](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201913649329503&dbt=NART)
- 인간 행동 분석을 이용한 위험 상황 인식 시스템 구현 [[링크]](http://koreascience.or.kr/article/JAKO202111841186714.page)
- 안전보건공단 - [한국산업안전보건공단 | 자료마당 | KOSHA Guide | KOSHA Guide | 분야별 안전보건기술지침 | 건설안전지침(C)](https://kosha.or.kr/kosha/data/guidanceC.do)

참고 튜토리얼:

*MediaPipe 활용 관련*

- [Body Language Detector 만들기](https://www.youtube.com/watch?v=We1uB79Ci-w&list=PLY-IV1aw5rP5kLQogj-2MmOeHL5tP0-yf&index=6)
- [신체 관절 간 각도 계산하기](https://www.youtube.com/watch?v=06TE_U21FK4)
