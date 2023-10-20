1. 결정트리를 학습할 때 쓰는 함수?
- DecisionTreeClassifier(max_depth=2, random_state=42)
- 최대 가지수를 2개로 해서 학습
2. gini의 값이 무엇을 의미하는지?
- 불순도를 의미한다.
- 1에 가까울수록 정확도가 낮아진다.
- 계산 방법은 분류해낸 붓꽃의 품종 갯수, 만약 2번째 붓꽃일 떄 value[0,49,5]라면 (0/54),(49/54),(5/54) 를 제곱해 1에서 빼준다.
3. 가지가 많아 질 때 gini의 값이 어떻게 변하는지?
- gini의 값이 0에 가까워 지도록 지정한 가지 수 만큼 가지를 늘려가면서 학습한다.
4. 결정트리의 가지가 많을 수록 좋은가?
- 결정트리의 가지가 많을 수록 gini 불순도가 낮아지고 좀 더 정확히 분류하기 때문에 좋을 것이다. 단점은 있는가?
5. gini와 entropy의 차이점(계산방식)과 무엇을 사용하면 좋은지??
- gini는 1의 값에서 빼는 것이라면 entropy는 클래스에 속해있는 샘플을 가지고 이진트리를 통해 계산하게 된다.
- gini가 계산이 더 빠르기 때문에 기본값으로 좋다. 값은 큰 차이 없이 둘다 비슷한 트리를 만든다.
- 다른 트리가 만들어지는 경우 지니 불순도가 가장 빈도 높은 클래스를 한쪽 가지로 고립시키는 경향이 있는 반면 엔트로피는 조금 더 균형 잡힌 트리를 만든다.
6. 규제가 없을 때 어떻게 분류되나?
- 데이터 하나하나 다 찾아서 분류 하려고 하기 때문에 과적합이 될 수 있다.
7. 규제의 종류에는 무엇이 있나?
- max_depth: 결정 트리의 최대 높이 제한
- min_samples_split: 노드를 분할하기 위해 필요한 최소 샘플 수
- min_samples_leaf: 리프 노드가 가지고 있어야 할 최소 샘플 수
- min_weight_fraction_leaf: 
  - 샘플 별로 가중치가 설정된 경우: 가중치의 전체 합에서 해당 리프 노드에 포함된 샘플의 가중치의 합이 차지하는 비율
  - 샘플 별로 가중치가 없는 경우: min_samples_leaf와 동일한 역할 수행
- max_leaf_nodes: 허용된 리프 노드의 최대 개수
- max_features: 각 노드에서 분할 평가에 사용될 수 있는 최대 특성 수
- max_: 매개변수를 감소
- min_: 매개변수를 증가
8. 데이터 분류를 잘 하려면?
- 적절한 규제 매개 변수를 골라 하이퍼파라미터의 값을 조절 하면서 찾아야 한다.
9. 회귀에서 결정 트리를 사용하려면?
- DecisionTreeRegressor을 사용한 결정트리 회귀 모델을 사용한다
10. 회귀모델에서의 값들?
- value 값은 노드의 속한 훈련샘플의 평균값을 나타낸다.
- mse는 gini와 같은 역할로 훈련 샘플의 평균제곱오차를 나타낸다.
11. 회귀모델에서의 규제?
- 분류에서와 같이 규제가 없는 경우 과대적합이 발생하고 적당한 값으로 규제할 경우 원하는 결과를 나타낼수 있다.
12. 결정트리의 단점? (4번과 연결)
- 훈련세트에 매우 민감하다.
  - 계단 형태: 훈련 세트의 회전에 매우 민감하다. ex) 훈련세트를 특징이 잘 들어나도록 형태를 바꾸는 경우
- 작은 변화에도 매우 민감하다.
  - 데이터에서 하나의 샘플을 제거 했을 때 다르게 학습할 수 있다.

============================== 연습문제 7번 ==============================

다음 단계를 따라 moons 데이터셋에 결정트리를 훈련시키고 세밀하게 튜닝하라
1. make_moons(n_sample=1000, noise=0.4)를 사용해 데이터셋을 생성한다
2. 이를 train_test_split()을 사용해 훈련세트와 데이터 세트로 나눈다
3. DecisionTreeClassifier의 최적의 매개변수를 찾기 위해 교차 검증과 함께 그리드 탐색을 수행한다
(GridSearchCV를 사용하면 됨. 여러가지 max_leaf_nodes 값을 시도)
4. 잦은 매개변수를 사용해 전체 훈련 세트에 대해 모델을 훈련시키고 테스트 세트에서 성능을 측정한다.
대략 85~87%의 정확도가 나옴

![image](https://github.com/Ourumo/ML/assets/133006490/fd089a7a-c284-4628-bb60-e0ae92dbacfe)
- 우선 코드를 시작하기 전에 기본 세팅을 해준다.

![image](https://github.com/Ourumo/ML/assets/133006490/6c07850f-07f7-4080-bef8-9a57135ed41b)
- 초승달 모양 클러스터 두 개 형상의 데이터를 생성
  - n_samples: 샘플 수
  - noise: 가우시안 노이즈의 표준 편차
  - random_state: 결과를 일정하게 해주기 위한 값 (보편적으로 42를 많이 사용)

![image](https://github.com/Ourumo/ML/assets/133006490/9aee9821-aaab-44f1-922a-39d84f3b5fbb)
- 훈련 세트와 테스트 세트 분리
  - test_size: 테스트 사이즈의 비율로, 코드에서는 8:2로 분리함 (훈련 8 : 테스트 2)

![image](https://github.com/Ourumo/ML/assets/133006490/b5b81c3e-217f-41b1-b455-7b13ea6c7ee0)
- 가장 좋은 매개변수를 찿기 위해 GridSearchCV를 임포트
  - max_leaf_nodes: 트리 구조에서 리프 노드의 최대 갯수
  - min_samples_split: 노드를 분할하기 위한 최소한의 샘플 갯수
  - verbose: 메시지 출력
    - verbose=0(default)이면 메시지 출력 안함
    - verbose=1이면 간단한 메시지 출력
    - verbose=2이면 하이퍼 파라미터별 메시지 출력
  - cv: 교차 검증 횟수
  - grid_search_cv.fit을 통해 학습 진행

![image](https://github.com/Ourumo/ML/assets/133006490/8c241a59-6854-495f-b959-0d14fb62cbf9)
- 294(max_leaf_nodes의 크기 * min_samples_split 크기) * 3(cv의 크기) = 892
- 892번의 fits를 진행

![image](https://github.com/Ourumo/ML/assets/133006490/b461e637-1430-4955-87dc-8cdee1de2dcc)

![image](https://github.com/Ourumo/ML/assets/133006490/f73d4f72-287e-4958-b14f-f81c13712a51)
- best_estimator: 가장 좋은 성능을 가지는 모델
  - max_leaf_nodes = 17

![image](https://github.com/Ourumo/ML/assets/133006490/964be33f-1500-4177-bb58-97a49b5c1e99)
- best_index: best_estimator로 선택된 모델의 인덱스 번호 
- best_params: best_estimator로 선택된 모델의 파라미터 값
  - max_leaf_nodes = 17
  - min_samples_split = 2
- best_score: best_estimator로 선택된 모델의 정확도
  - 85% 정도?

![image](https://github.com/Ourumo/ML/assets/133006490/e0c82f75-2cd3-4087-849a-ff7918570854)
- 전체 훈련 세트로 찾은 최적의 모델을 다시 훈련하고 성능 측정

![image](https://github.com/Ourumo/ML/assets/133006490/c7698ad7-faf1-4e27-b773-2fe03d30e121)
- 성능으로 (0.8695)가 나옴
- 위에서 찾은 성능 즉, 정확도가 서로 다른 이유?
  - 8:2로 훈련세트와 테스트 세트를 분리하는 등의 학습과 검증 과정을 수행하면서 정확도의 차이가 발생함
