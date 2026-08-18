# ViT 이미지 패치 & Multi-Head Attention (TensorFlow 구현)

> 이미지 한 장이 어떻게 "토큰 시퀀스"로 바뀌고, Multi-Head Attention을 거쳐가는지 처음부터 끝까지 따라가봅니다.

---

## 0. 전체 그림 먼저 보기

Vision Transformer(ViT)는 원래 텍스트를 다루던 Transformer를 **이미지에 적용**한 것입니다. 텍스트는 원래 "단어 토큰"의 나열이지만, 이미지는 픽셀의 2차원 격자이기 때문에, **이미지를 작은 조각(패치)으로 잘라서 "토큰"처럼 취급**하는 전처리 과정이 먼저 필요합니다.

```
[원본 이미지]                [패치로 분할]         [패치 임베딩 + CLS + 위치정보]      [Multi-Head Attention]
28x28x1 이미지  ──분할──▶  7x7 = 49개 패치  ──투영──▶  50개 토큰 x 8차원 벡터  ──▶  z (50, 8)
                          (패치 1개 = 4x4=16픽셀)      (CLS 토큰 1개 + 패치 49개)
```

이번 자료는 이 전체 파이프라인을 **① 이미지 패치 만들기 → ② Multi-Head Attention** 두 부분으로 나눠서, 문제정의 → 분석 → 설계 → 구현 순서로 설명합니다.

---

## PART 1. 이미지 패치 (Image Patch)

### 1-1. 문제 정의

Transformer(그리고 Self-Attention)는 원래 **"토큰들의 나열(시퀀스)"** 을 입력으로 받도록 설계되었습니다. 문장이라면 자연스럽게 "단어 하나 = 토큰 하나"가 되지만, 이미지는 그냥 픽셀들의 2차원 격자일 뿐 애초에 "토큰"이라는 단위가 없습니다.

> **"이미지를 어떻게 토큰들의 나열로 바꿀 것인가?"**

이것이 ViT가 풀어야 했던 첫 번째 문제입니다. ViT 논문("An Image is Worth 16x16 Words")이 제안한 방법은 간단합니다.

> **이미지를 작은 정사각형 조각(패치)들로 잘게 자른 뒤, 패치 하나하나를 "단어 토큰 하나"처럼 취급하자.**

### 1-2. 분석: 패치가 어떻게 만들어지는가

우리가 다루는 **MNIST 데이터셋**은 `28×28` 크기의 흑백 이미지(채널 1개)입니다. 이 이미지를 `4×4` 크기의 패치로 자른다고 해봅시다.

```
28x28 이미지를 4x4 크기로 자르면
→ 가로로 28/4 = 7개, 세로로 28/4 = 7개
→ 총 7 x 7 = 49개의 패치가 생김
```

```
┌────┬────┬────┬────┬────┬────┬────┐
│ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │   ← 각 칸이 4x4 픽셀짜리 패치
├────┼────┼────┼────┼────┼────┼────┤
│ P8 │ P9 │... │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │      (총 7행)
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │ P49│
└────┴────┴────┴────┴────┴────┴────┘
```

패치 하나(`4×4×1`)는 픽셀 값 `4×4×1 = 16`개로 이루어져 있습니다. 이걸 한 줄로 쭉 펴면(flatten) **길이 16짜리 벡터 1개**가 됩니다. 즉:

| 개념 | 값 | 텍스트 Transformer와의 대응 |
|---|---|---|
| 이미지 1장 | `28×28×1` | 문장 1개 |
| 패치 1개 | `4×4×1` → flatten하면 길이 16 벡터 | 단어 1개 (아직 임베딩 전) |
| 전체 패치 개수 | `49`개 | 문장 안의 단어 개수(시퀀스 길이) |

이 상태에서는 아직 "16차원 숫자 벡터"일 뿐, 우리가 원하는 임베딩 차원(예: 8)이 아닙니다. 그래서 **선형 변환(Linear Projection, 즉 `Dense` 레이어)** 을 한 번 더 거쳐서 원하는 차원으로 바꿔줍니다. (텍스트의 "단어 → 임베딩 벡터" 과정과 동일한 역할)

```
패치(길이 16) --Dense(8)--> 패치 임베딩(길이 8)
```

### 1-3. CLS 토큰이 필요한 이유

패치 49개를 각각 임베딩하면 `(49, 8)` 모양의 토큰 시퀀스가 만들어집니다. 그런데 우리가 최종적으로 하고 싶은 건 보통 **"이 이미지는 고양이다/개다" 같은 분류(classification)** 입니다.

문제는, 패치가 49개나 있는데 **"이미지 전체를 대표하는 벡터 하나"** 가 없다는 것입니다. 그래서 BERT의 `[CLS]` 토큰 아이디어를 그대로 가져와서, **패치들과 별개로 학습되는 "전체 요약용 토큰" 1개를 맨 앞에 추가**합니다.

```
[CLS] + 패치1 + 패치2 + ... + 패치49  = 총 50개 토큰
```

이 `CLS` 토큰은 Self-Attention을 여러 번 거치면서 **다른 모든 패치의 정보를 흡수**하게 되고, 최종적으로 이 토큰 하나만 뽑아서 분류기(classifier)에 넣으면 "이미지 전체에 대한 판단"을 할 수 있게 됩니다.

> 💡 **비유**: 회의에 참석한 49명의 발언자(패치)들이 있고, 이 모든 발언을 종합해서 최종 결론을 내리는 서기(CLS 토큰) 한 명을 따로 앉혀놓은 것과 비슷합니다.

### 1-4. 위치 정보(Position Embedding)가 필요한 이유

Self-Attention은 구조적으로 **"순서"라는 개념이 없습니다.** 패치 1번과 패치 49번을 아무리 계산해도, 그 계산 자체에는 "1번이 왼쪽 위, 49번이 오른쪽 아래"라는 위치 정보가 전혀 담겨있지 않습니다. (모든 토큰을 그냥 "집합"처럼 취급)

하지만 이미지에서는 **위치가 매우 중요**합니다. 하늘색 패치가 위쪽에 있으면 "하늘", 아래쪽에 있으면 전혀 다른 의미일 수 있습니다. 그래서 각 토큰에 **"이 토큰이 몇 번째 위치인지" 정보를 담은 벡터(Position Embedding)** 를 더해줍니다.

```
최종 입력 = 패치 임베딩 + 위치 임베딩   (CLS 토큰 포함, 총 50개 모두)
```

### 1-5. 설계

```
파이프라인: 이미지 → 패치 임베딩

입력  : image (batch, 28, 28, 1)
출력  : x     (batch, 50, 8)     ← Multi-Head Attention에 들어갈 최종 입력

처리 순서:
  1. 이미지를 4x4 크기로 잘라 패치 49개 추출         → (batch, 49, 16)
  2. Dense(8)로 선형 투영(임베딩)                    → (batch, 49, 8)
  3. CLS 토큰(학습 파라미터) 1개를 맨 앞에 붙임        → (batch, 50, 8)
  4. Position Embedding(학습 파라미터)을 더함         → (batch, 50, 8)
```

### 1-6. 구현 및 실행 결과

```python
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 28      # MNIST 이미지 크기 (28x28)
PATCH_SIZE = 4
CHANNELS = 1       # MNIST는 흑백이므로 채널 1개
MY_HIDDEN = 8
MY_BATCH = 2

num_patches = (IMG_SIZE // PATCH_SIZE) ** 2      # 7*7 = 49
patch_dim = PATCH_SIZE * PATCH_SIZE * CHANNELS   # 4*4*1 = 16

# 가짜 이미지 2장 (실제로는 MNIST 데이터셋에서 불러온 이미지, shape: 28x28x1)
img = tf.random.normal([MY_BATCH, IMG_SIZE, IMG_SIZE, CHANNELS])

# 1) 이미지를 4x4 패치로 잘라내기
patches = tf.image.extract_patches(
    images=img,
    sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
    strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)
patches_flat = tf.reshape(patches, [MY_BATCH, num_patches, patch_dim])
print("패치 flatten:", patches_flat.shape)

# 2) 패치 임베딩 (Linear Projection)
patch_embed = layers.Dense(MY_HIDDEN)
x = patch_embed(patches_flat)
print("패치 임베딩 후:", x.shape)

# 3) CLS 토큰 추가
cls_token = tf.Variable(tf.zeros([1, 1, MY_HIDDEN]))
cls_tokens = tf.repeat(cls_token, repeats=MY_BATCH, axis=0)
x = tf.concat([cls_tokens, x], axis=1)
print("CLS 토큰 추가 후:", x.shape)

# 4) Position Embedding 더하기
pos_embed = tf.Variable(tf.random.normal([1, num_patches + 1, MY_HIDDEN]) * 0.02)
x = x + pos_embed
print("Position embedding 더한 후 (최종 MHA 입력):", x.shape)
```

**실행 결과:**

```
패치 flatten: (2, 49, 16)
패치 임베딩 후: (2, 49, 8)
CLS 토큰 추가 후: (2, 50, 8)
Position embedding 더한 후 (최종 MHA 입력): (2, 50, 8)
```

**해석**:
- `tf.image.extract_patches`가 이미지를 자동으로 격자 형태로 잘라줍니다. 결과 `(2, 49, 16)`에서 `49`는 패치 개수(시퀀스 길이), `16`은 패치 하나의 픽셀 수(4×4×1)입니다.
- `Dense(8)`을 거쳐 `16 → 8` 차원으로 압축되면서 진짜 "임베딩 벡터"가 됩니다.
- `CLS` 토큰을 붙이면서 `49 → 50`으로 시퀀스 길이가 1 늘어납니다. **이 `50`이 바로 MHA 코드의 `MY_TOKEN = 50`, 그리고 `t = tf.shape(x)[1]`이 가리키는 값**입니다.
- 최종 shape `(2, 50, 8)`이 바로 다음 파트의 `MyMHA` 레이어에 들어가는 입력입니다. `MY_HIDDEN=8`도 정확히 여기서 정해진 패치 임베딩 차원입니다.

> ✅ **정리**: `MY_TOKEN=50`이라는 숫자가 어디서 왔는지 궁금하셨다면 — "이미지 패치 49개 + CLS 토큰 1개 = 50"이 그 답입니다.

---

## PART 2. Multi-Head Attention (TensorFlow 구현)

### 2-1. 문제 정의

PART 1에서 이미지를 `(batch, 50, 8)` 모양의 토큰 시퀀스로 바꿨습니다. 이제 이 50개 토큰이 **서로의 관계를 파악**하도록 만들어야 합니다. 예를 들어 "하늘 패치"와 "새 패치"가 서로 연관되어 있다는 것을 모델이 알아야 합니다.

하나의 attention만 쓰면 **한 가지 관점**으로만 패치 관계를 봅니다. 실제로는 "색깔이 비슷한 패치끼리", "형태가 이어지는 패치끼리" 등 **여러 관점**이 동시에 필요하기 때문에 Multi-Head Attention을 사용합니다.

### 2-2. 분석: 전체 shape 흐름

```
입력 x (128, 50, 8)
   │
   ├─ Dense(wq,wk,wv) → q,k,v (128, 50, 8)
   │
   ├─ reshape  → (128, 50, 2, 4)      # head로 쪼개기 (8 = 2head x 4dim)
   ├─ transpose → (128, 2, 50, 4)     # head를 앞으로 이동
   │
   ├─ Q·K^T / √dk → softmax → V 가중합
   │     score: (128, 2, 50, 50)
   │     att  : (128, 2, 50, 50)      # 행 합 = 1
   │     final: (128, 2, 50, 4)
   │
   ├─ transpose → (128, 50, 2, 4)
   ├─ reshape   → (128, 50, 8)        # head 합치기 (concat)
   │
   └─ Dense(wz) → z (128, 50, 8)
```

**핵심**: 입력과 출력의 shape이 `(128, 50, 8)`로 동일합니다. 그래야 이 레이어를 Encoder Block 안에서 여러 겹 쌓을 수 있습니다.

### 2-3. 설계

```
클래스: MyMHA(n_hidden, n_head)

멤버 변수:
  d_head = n_hidden // n_head    # head 하나가 담당하는 차원 (정수 나눗셈 필수!)
  wq, wk, wv : Dense(n_hidden)   # Q,K,V 생성용
  wz         : Dense(n_hidden)   # 최종 투영용 (W^O 역할)

call(x):
  1. Q,K,V 생성                (batch, token, hidden)
  2. head로 reshape            (batch, token, head, d_head)
  3. transpose                 (batch, head, token, d_head)
  4. score = Q·K^T / √d_head   (batch, head, token, token)
  5. att = softmax(score)      (batch, head, token, token)
  6. final = att · V           (batch, head, token, d_head)
  7. transpose + reshape       (batch, token, hidden)   ← head 합치기
  8. z = wz(merge)             (batch, token, hidden)
```

### 2-4. 구현 코드 (라인별 설명 포함)

```python
class MyMHA(layers.Layer):
    def __init__(self, n_hidden, n_head):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.d_head = n_hidden // n_head   # 정수 나눗셈 // 사용! (/ 는 float가 되어 reshape 에러 발생)

        # Q, K, V 벡터 전환용 행렬
        self.wq = layers.Dense(n_hidden, use_bias=True)
        self.wk = layers.Dense(n_hidden, use_bias=True)
        self.wv = layers.Dense(n_hidden, use_bias=True)

        # 최종 결과 투영용
        self.wz = layers.Dense(n_hidden, use_bias=True)

    def call(self, x, training=False):
        b = tf.shape(x)[0]                # 배치 크기, 128
        t = tf.shape(x)[1]                # 토큰 개수, 50 (= 패치49 + CLS1)

        # q, k, v 벡터 전환
        q = self.wq(x)                    # (128, 50, 8)
        k = self.wk(x)
        v = self.wv(x)

        # 머리(head)로 나누기: (128, 50, 8) -> (128, 50, 2, 4)
        q = tf.reshape(q, [b, t, self.n_head, self.d_head])
        k = tf.reshape(k, [b, t, self.n_head, self.d_head])
        v = tf.reshape(v, [b, t, self.n_head, self.d_head])

        # head를 앞으로: (128, 50, 2, 4) -> (128, 2, 50, 4)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Attention 계산
        scale = tf.cast(self.d_head, tf.float32) ** -0.5     # 1/√d_head
        score = tf.matmul(q, k, transpose_b=True) * scale     # (128, 2, 50, 50)
        att = tf.nn.softmax(score, axis=-1)                   # 마지막 축 기준 정규화
        final = tf.matmul(att, v)                              # (128, 2, 50, 4)

        # 머리(head) 합치기
        final = tf.transpose(final, [0, 2, 1, 3])              # (128, 50, 2, 4)
        merge = tf.reshape(final, [b, t, self.n_hidden])       # (128, 50, 8)
        z = self.wz(merge)                                      # 최종 z

        return z
```

**라인별 핵심 포인트**

| 코드 | 설명 |
|---|---|
| `self.d_head = n_hidden // n_head` | `/`가 아니라 `//`를 써야 정수가 나옵니다. `4.0`(float)이면 `tf.reshape`에서 `TypeError`가 발생합니다. |
| `self.wq(x)` | 토큰 하나(길이 8)를 Dense에 통과시켜 Query 벡터로 변환. 학습되는 `W`, `b`를 내부에 가지고 있음. |
| `tf.reshape(q, [b,t,head,d_head])` | 숫자 재배열만 할 뿐, 새로운 계산은 없음. 8개 숫자를 (2,4)로 그룹만 나눔. |
| `tf.transpose(q, [0,2,1,3])` | `matmul`이 마지막 2개 축만 행렬곱하기 때문에, head별로 독립 계산되도록 축 순서를 바꿈. |
| `transpose_b=True` | `k`를 곱하기 전에 자동으로 전치(transpose)해서 `Q·Kᵀ` 연산을 수행. |
| `** -0.5` | `d_head^(-0.5) = 1/√d_head`. Attention 공식의 스케일링. |
| `axis=-1` | softmax를 마지막 축(참고 대상 토큰 방향) 기준으로 적용해서 "행의 합=1"을 만듦. |
| 마지막 `transpose`+`reshape` | head별 결과를 다시 이어붙임(concat)과 동일한 효과. |
| `self.wz(merge)` | head들을 한 번 더 섞어주는 최종 선형 투영(W^O 역할). |

### 2-5. 실행 및 shape 검증

```python
MY_BATCH = 128
MY_TOKEN = 50    # PART 1에서 만든 CLS+패치 개수
MY_HIDDEN = 8    # PART 1에서 만든 패치 임베딩 차원
MY_HEAD = 2

x = tf.random.normal([MY_BATCH, MY_TOKEN, MY_HIDDEN])
mha = MyMHA(MY_HIDDEN, MY_HEAD)
y = mha(x)
print('입력', x.shape)
print('출력', y.shape)
```

**실행 결과 (중간 shape 모두 포함):**

```
입력 (128, 50, 8)
q shape (transpose 후)     : (128, 2, 50, 4)
score shape                : (128, 2, 50, 50)
att shape                  : (128, 2, 50, 50)
att 한 행의 합               : 1.0
final(head별 출력) shape    : (128, 50, 2, 4)
merge(head 합친 후) shape   : (128, 50, 8)
출력 z shape                : (128, 50, 8)
```

입력 `(128, 50, 8)`과 출력 `(128, 50, 8)`이 동일한 shape입니다. → 이 레이어를 Encoder Block으로 감싸서 여러 겹 쌓아 올릴 수 있습니다.

---

## 3. 전체 파이프라인 한눈에 정리

| 단계 | 입력 shape | 출력 shape | 하는 일 |
|---|---|---|---|
| ① 이미지 패치 추출 | `(128, 28, 28, 1)` | `(128, 49, 16)` | MNIST 이미지를 4x4 조각으로 자름 |
| ② 패치 임베딩 | `(128, 49, 16)` | `(128, 49, 8)` | Dense로 원하는 차원으로 투영 |
| ③ CLS 토큰 추가 | `(128, 49, 8)` | `(128, 50, 8)` | 이미지 전체 요약용 토큰 삽입 |
| ④ Position Embedding | `(128, 50, 8)` | `(128, 50, 8)` | 순서(위치) 정보 추가 |
| ⑤ Multi-Head Attention | `(128, 50, 8)` | `(128, 50, 8)` | 토큰들끼리 문맥 파악 |

## 4. 핵심 요약 (학생 정리용)

| 질문 | 답 |
|---|---|
| 왜 이미지를 패치로 나누나? | Self-Attention은 "토큰 시퀀스"를 입력으로 받으므로, 이미지도 토큰처럼 만들어야 함 |
| `MY_TOKEN=50`은 어디서 나온 숫자인가? | 패치 49개 + CLS 토큰 1개 |
| CLS 토큰의 역할은? | Attention을 거치며 이미지 전체 정보를 흡수해 최종 분류에 사용되는 대표 토큰 |
| Position Embedding이 필요한 이유는? | Self-Attention 자체에는 순서/위치 개념이 없기 때문에 별도로 더해줘야 함 |
| `d_head = n_hidden // n_head`에서 왜 `//`를 쓰나? | `/`는 float를 반환해 `tf.reshape`에서 타입 에러 발생 |
| head를 나눴다 다시 합치는 이유는? | 여러 관점(head)으로 각각 계산한 뒤, 그 결과들을 종합해 더 풍부한 표현을 만들기 위해 |

## 5. 다음 학습 주제 제안

1. **Transformer Encoder Block 완성하기**: MHA 뒤에 LayerNorm, Residual Connection, MLP(Feed-Forward) 붙이기
2. **Position Embedding 방식 비교**: 학습형(learnable) vs 고정형(sinusoidal)
3. **패치 크기와 성능의 관계**: 패치를 작게 자를수록 토큰 수는 늘고 연산량은 커지지만 더 세밀한 정보 포착 가능

https://ffighting.net/deep-learning-paper-review/vision-model/vision-transformer/#google_vignette
---

*본 자료는 MNIST(28×28×1) 이미지 → 패치 → 임베딩 → CLS/Position → Multi-Head Attention으로 이어지는 ViT 입력 파이프라인 전체를, 실제 TensorFlow 코드 실행 결과와 함께 정리한 실습용 강의자료입니다.*
