# Vision Transformer 구현 - 단계별 코드 설명

> 이 자료는 `myViT_fixed.py`를 **개념 → 수식/구조 → 코드 → 실행 결과** 순서로
> 한 단계씩 설명합니다. TDD 방식에 맞춰 각 단계마다 "이 코드가 무엇을 보장해야
> 하는가"를 먼저 정의하고, 실제 코드와 테스트 결과로 확인합니다.

---

## 전체 그림 먼저 보기

Vision Transformer는 문장을 다루던 Transformer 구조를 그대로 가져와서,
"단어" 대신 "이미지 조각(패치)"을 입력으로 사용합니다.

```
손글씨 이미지(28×28×1)
    │  ① 패치로 자르기                              [Step 2]
    ▼
패치 시퀀스 (49x16)
    │  ② Linear 임베딩                               [Step 3]
    ▼
패치 토큰 (49x8)
    │  ③ CLS 토큰 추가                                [Step 4]
    ▼
토큰 50개 (CLS 1개 + 패치 49개)(50x8)
    │  ④ 위치 임베딩 더하기                            [Step 5]
    ▼
Encoder 입력 z0 (50x8)
    │  ⑤ Transformer Encoder × 6회 반복                [Step 6, 7, 8]
    ▼
Encoder 출력 zL (50x8)
    │  ⑥ Global Average Pooling (50개 토큰 → 1개로 압축) [Step 9]
    ▼
Pooling 결과 (1x8)
    │  ⑦ Dense(10) + Softmax (분류 헤드)               [Step 10]
    ▼
클래스 확률 (1x10)
    ▼
숫자 0~9 중 하나
```

각 화살표 옆 대괄호 `[Step N]`은 아래 본문의 Step 번호와 정확히 대응됩니다.
⑤번 화살표는 세 개의 Step으로 이루어져 있다는 점에 주의하세요:
**Step 6**(MHSA 부품 자체를 만드는 방법), **Step 7**(그 MHSA를 정규화·잔차연결과
묶어 Encoder 블록 하나로 만드는 방법), **Step 8**(그 Encoder 블록을 6번
반복해서 실제로 쌓는 방법)이 합쳐져 ⑤번 화살표 하나가 됩니다.

괄호 안 숫자는 "토큰 개수 × 임베딩 차원" 모양을 뜻하며, 배치 크기(B=128)는
생략하고 표기했습니다. 예를 들어 `(49x16)`은 실제로는 `(128, 49, 16)`에서
배치 차원을 뺀 모양입니다.

이전 버전에서는 ⑥번(Pooling)과 ⑦번(분류 헤드)을 하나로 묶어 "대표 토큰만
뽑아 분류"라고 뭉뚱그렸는데, 이번 버전에서는 이 둘을 **서로 다른 연산이므로
Step도 분리**했습니다. Pooling은 "50개 토큰을 1개로 압축"하는 연산이고,
분류 헤드는 "압축된 1개 벡터를 10개 클래스 확률로 바꾸는" 연산으로 역할이
다릅니다.

이 순서를 코드에서 그대로 단계별로 구현했습니다. 아래에서 하나씩 뜯어봅니다.


<img width="1200" height="672" alt="image" src="https://github.com/user-attachments/assets/6c9cb052-8b2b-4313-8043-2ca4a6d82de4" />


<img width="814" height="741" alt="image" src="https://github.com/user-attachments/assets/06ba7e97-5f53-4d67-a162-e30afa709df0" />


---

## Step 1. 데이터 준비 — "학습 재료를 만드는 단계"

**개념**
모델은 0과 1 사이의 실수만 이해할 수 있고, 입력 모양이 반드시 `(28, 28, 1)`
(높이, 너비, 채널)이어야 합니다. MNIST 원본 데이터는 0~255 정수이고 채널
차원도 없기 때문에 전처리가 필요합니다.

**주의할 점 (이전 버전의 버그)**
학습 데이터(`x_train`)만 전처리하고 평가 데이터(`x_test`)를 빠뜨리면,
학습은 되지만 평가 단계(`model.evaluate`)에서 모양이 안 맞아 에러가 납니다.
→ **두 데이터셋은 항상 "같은 규칙"으로 함께 전처리**해야 합니다.

**코드**
```python
x_train = (x_train.astype('float32') / 255.0)[..., None]
x_test = (x_test.astype('float32') / 255.0)[..., None]

assert x_train.shape[1:] == (28, 28, 1)
assert x_test.shape[1:] == (28, 28, 1)
```

**실행 결과 예시**
```
[검증] x_train 모양: (60000, 28, 28, 1)  값 범위: 0.0 ~ 1.0
[검증] x_test 모양 : (10000, 28, 28, 1)  값 범위: 0.0 ~ 1.0
```

---

## Step 2. 이미지 패치화 — "이미지를 단어처럼 자르기" *(흐름도 ①)*

**개념**
28×28 이미지를 4×4 크기 조각 49개(=7×7)로 나눕니다. 원 논문 표기로는
이미지 크기 `(H, W, C)`, 패치 크기 `P`일 때 패치 개수는 다음과 같습니다.

```
N = (H × W) / (P × P)
```

이 코드에서는 `H=W=28`, 패치 한 변 크기 `4` (=28÷7) 이므로
`N = 28×28 / (4×4) = 49`개의 패치가 만들어집니다. 각 패치는 4×4=16개의
픽셀값을 일렬로 편 벡터가 됩니다.

**코드**
```python
def PatchLayer():
    size = 28 // MY_PATCH      # 패치 한 변 크기 = 4
    num = MY_PATCH * MY_PATCH  # 패치 개수 = 49
    dim = size * size          # 패치 벡터 차원 = 16

    def extract(x):
        p = tf.image.extract_patches(
            images=x,
            sizes=[1, size, size, 1],
            strides=[1, size, size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        p = tf.reshape(p, [-1, num, dim])
        return p

    return layers.Lambda(extract)
```

**실행 결과 예시**
```
입력 (128, 28, 28, 1)
출력 (128, 49, 16)
```
128은 배치 크기(한 번에 처리하는 이미지 수), 49는 패치 개수, 16은 패치 하나의 픽셀 수입니다.

---

## Step 3. 패치 임베딩 — "픽셀 값을 의미 있는 벡터로 변환" *(흐름도 ②)*

**개념**
16차원짜리 "생 픽셀값" 벡터는 아직 아무 의미도 없습니다. `Dense` 층(=행렬 곱)을
거쳐 모델이 학습으로 의미를 부여할 수 있는 8차원 벡터로 바꿔줍니다. 이 변환
행렬을 논문에서는 `E`라고 부르며, 크기는 (패치벡터차원 → 임베딩차원)입니다.

**코드**
```python
tokens = layers.Dense(MY_HIDDEN)(patches)   # 16차원 -> 8차원
```

**실행 결과 예시**
```
패치화 후 (None, 49, 16)
임베딩 후 (None, 49, 8)
```

---

## Step 4. CLS 토큰 추가 — "정답을 대표할 토큰 하나 준비" *(흐름도 ③)*

**개념**
49개의 패치 토큰만으로는 "이미지 전체를 대표하는 하나의 결과"를 얻기 애매합니다.
그래서 원 논문은 이미지 내용과 무관하게 **처음부터 학습되는 벡터 하나**를
맨 앞에 붙여두고, Encoder를 통과하면서 이 토큰이 "전체 이미지의 요약 정보"를
흡수하도록 학습시킵니다. 이 토큰을 최종 분류에 사용합니다.

**주의할 점 (이전 버전의 설계 문제)**
CLS 토큰을 "패치들의 평균값을 변환한 것"으로 만들면, 이 토큰이 이미
입력 내용에 종속되어 시작하게 되어 논문의 의도(독립적으로 학습되는 대표 토큰)와
달라집니다. 아래처럼 입력과 무관한 파라미터로 만드는 것이 표준 방식입니다.

**코드**
```python
class ClsToken(layers.Layer):
    def build(self, input_shape):
        self.cls = self.add_weight(
            shape=(1, 1, self.n_hidden),
            initializer='random_normal',
            trainable=True
        )
    def call(self, x):
        b = tf.shape(x)[0]
        return tf.tile(self.cls, [b, 1, 1])   # 배치 크기만큼 복제
```

**실행 결과 예시**
```
클래스 토큰 추가 후 (None, 50, 8)   # 49개 패치 + CLS 토큰 1개 = 50
```

---

## Step 5. 위치 임베딩 — "패치가 어디에 있었는지 알려주기" *(흐름도 ④)*

**개념**
Attention 연산은 순서/위치 정보를 모릅니다 (모든 토큰을 동시에 봄). 그래서
sin/cos 함수로 각 위치마다 고유한 패턴(지문 같은 것)을 만들어 토큰에
더해줍니다. 위치 `i`, 임베딩 차원 `j`에 대해:

```
짝수 차원:  PE(i, j) = sin( i / 10000^(j/d) )
홀수 차원:  PE(i, j) = cos( i / 10000^((j-1)/d) )
```

**코드**
```python
def pos_embed(n_token, d_hidden):
    pe = np.zeros((n_token, d_hidden), dtype=np.float32)
    for i in range(n_token):
        for j in range(d_hidden):
            if j % 2 == 0:
                pe[i, j] = np.sin(i / 10000 ** (j / d_hidden))
            else:
                pe[i, j] = np.cos(i / 10000 ** ((j - 1) / d_hidden))
    return tf.constant(pe, dtype=tf.float32)

x = tokens + pe   # 토큰에 위치 정보를 더함
```

**실행 결과 예시**
```
위치 임베딩 후 (None, 50, 8)
```

---

## Step 6. Multi-Head Self-Attention — "서로 얼마나 관련 있는지 계산" *(흐름도 ⑤의 부품)*

**개념**
각 토큰이 Query(질문), Key(열쇠), Value(값) 세 가지 벡터를 만들고,
"내 Query가 상대방 Key와 얼마나 잘 맞는지"를 점수로 계산해서 그 점수만큼
상대방의 Value를 가져옵니다. 이 계산을 여러 "머리(head)"로 나눠서
동시에 서로 다른 관점으로 수행합니다.

```
score = (Q · Kᵀ) / √dₖ
Attention = softmax(score) · V
```

`dₖ`는 머리 하나가 담당하는 차원 수이며, 이 코드에서는
`MY_HIDDEN(8) ÷ MY_HEAD(2) = 4`입니다.

**코드**
```python
class MyMHA(layers.Layer):
    def call(self, x, training=False):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)          # (B,50,8)
        q = tf.reshape(q, [b, t, self.n_head, self.d_head])   # 머리로 나누기
        ...
        scale = tf.cast(self.d_head, tf.float32) ** -0.5
        score = tf.matmul(q, k, transpose_b=True) * scale     # (B,2,50,50)
        att = tf.nn.softmax(score, axis=-1)
        final = tf.matmul(att, v)                              # (B,2,50,4)
        ...
        return self.wz(merge)                                  # (B,50,8)
```

**실행 결과 예시**
```
입력 (128, 50, 8)
출력 (128, 50, 8)   # 모양은 그대로, 내용은 서로의 정보를 반영해 바뀜
```

---

## Step 7. Encoder 블록 — "정규화 + Attention + MLP를 하나로 묶기" *(흐름도 ⑤의 부품)*

**개념**
Attention 하나만으로는 표현력이 부족해서, 다음 두 서브블록을 순서대로
쌓습니다. 이때 원 논문(ViT)은 일반 Transformer와 달리 **연산 전에 먼저
정규화(Pre-Norm)** 를 하고, 입력을 결과에 다시 더해주는 잔차 연결
(Residual Connection)을 사용합니다.

```
z' = z + MSA(LN(z))
z'' = z' + MLP(LN(z'))
```

**주의할 점 (이전 버전의 버그)**
`LN`이 어텐션 앞과 MLP 앞 두 군데에서 쓰이는데, **같은 레이어 객체를
재사용하면 두 지점이 파라미터(γ, β)를 공유**하게 되어 위 수식과 달라집니다.
반드시 독립된 두 개의 LayerNorm이어야 합니다.

**코드**
```python
def encoder_block(x, n_hidden, n_head, mlp_mult):
    norm1 = layers.LayerNormalization(epsilon=1e-5)   # 어텐션 전용
    norm2 = layers.LayerNormalization(epsilon=1e-5)   # MLP 전용

    y = norm1(x)
    y = MyMHA(n_hidden, n_head)(y)
    y = layers.Add()([x, y])          # 잔차 연결 1

    w = norm2(y)
    expand = mlp_mult * n_hidden      # 10 × 8 = 80차원으로 확장
    y = layers.Dense(expand, activation=tf.nn.gelu)(w)
    y = layers.Dense(n_hidden)(y)     # 다시 8차원으로 축소
    y = layers.Add()([w, y])          # 잔차 연결 2
    return y
```

**실행 결과 예시**
```
MLP 확장 차원 실제값: 80
입력 (128, 50, 8)
출력 (128, 50, 8)
```

---

## Step 8. 모델 조립 — "패치화부터 Encoder까지 순서대로 연결" *(흐름도 ①~⑤ 전체 연결)*

**개념**
Step 2~7에서 만든 부품(패치화, 임베딩, CLS 토큰, 위치 임베딩, Encoder 블록)을
실제 순서대로 이어붙입니다. Encoder 블록은 `MY_ENCODER=6`번 반복해서 쌓습니다.
이 단계의 결과는 아직 분류 확률이 아니라, **50개 토큰이 서로의 정보를 반영한
상태의 텐서**입니다 (위 전체 그림의 "Encoder 출력 zL").

**코드**
```python
input = keras.Input(shape=MY_SHAPE)
patches = PatchLayer()(input)                              # ① (49,16)
tokens = layers.Dense(MY_HIDDEN)(patches)                   # ② (49,8)
ctoken = ClsToken(MY_HIDDEN)(tokens)
tokens = layers.Concatenate(axis=1)([ctoken, tokens])       # ③ (50,8)
x = tokens + pos_embed(MY_TOKEN, MY_HIDDEN)                 # ④ (50,8)

for _ in range(MY_ENCODER):                                 # ⑤ Encoder x6
    x = encoder_block(x, MY_HIDDEN, MY_HEAD, MY_MLP)
```

**실행 결과 예시**
```
패치화 후 (None, 49, 16)
임베딩 후 (None, 49, 8)
클래스 토큰 추가 후 (None, 50, 8)
위치 임베딩 후 (None, 50, 8)
인코더 추가 후 (None, 50, 8)
```

---

## Step 9. Global Average Pooling — "50개 토큰을 1개 대표값으로 압축" *(흐름도 ⑥)*

**개념**
Encoder 출력은 아직 토큰 50개짜리 텐서 `(50, 8)`입니다. 분류를 하려면
이걸 이미지 하나를 대표하는 벡터 1개로 줄여야 합니다. `GlobalAveragePooling1D`는
50개 토큰의 값을 **차원별로 평균**내어 `(1, 8)` 벡터 하나로 압축합니다.

```
pooled[j] = (1/50) × Σ (token_i[j])   for i = 1 .. 50
```

**주의할 점 (Step 7과의 차이)**
원 논문은 평균 대신 **CLS 토큰(0번째 토큰)만 그대로 뽑아서** 사용합니다
(`x[:, 0]`). 이 코드는 50개 토큰 전체를 평균 내는 방식을 쓰는데, 두 방식
모두 통용되지만 서로 다른 설계 선택이라는 점을 구분해서 이해해야 합니다.

**코드**
```python
x = layers.GlobalAveragePooling1D()(x)   # (50,8) -> (1,8)
```

**실행 결과 예시**
```
GAP 처리 후 (None, 8)
```

---

## Step 10. 분류 헤드 — "대표 벡터를 10개 클래스 확률로 변환" *(흐름도 ⑦)*

**개념**
Pooling으로 얻은 `(1, 8)` 벡터를 `Dense(10)` 층에 통과시켜 숫자 0~9 각각에
대한 점수를 만들고, `softmax`로 그 점수들을 "합이 1인 확률"로 바꿉니다.
가장 확률이 높은 클래스가 모델의 최종 예측이 됩니다.

```
logits = pooled · W + b        # (1,8) -> (1,10)
확률   = softmax(logits)        # 합이 1이 되도록 정규화
```

**코드**
```python
output = layers.Dense(MY_CLASS, activation='softmax')(x)   # (1,8) -> (1,10)
model = keras.Model(input, output)
```

**실행 결과 예시**
```
최종 출력 (None, 10)
```

이 `(None, 10)` 벡터가 위 전체 그림의 "클래스 확률 (1x10)"에 해당하며,
이 중 값이 가장 큰 위치의 인덱스(0~9)가 곧 "숫자 0~9 중 하나"라는 최종
예측값입니다.

---

## Step 11. 학습 및 평가

**개념**
분류 문제이므로 `sparse_categorical_crossentropy` 손실 함수를 쓰고,
`Adam` 옵티마이저로 5 epoch 학습한 뒤 평가 데이터로 정확도를 확인합니다.

**코드**
```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=MY_LEARN),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, epochs=MY_EPOCH)
loss, acc = model.evaluate(test_ds)
```

**실행 결과 예시**
```
학습 시작
Epoch 1/5  469/469 [====] - loss: 1.xx - accuracy: 0.6x
...
Epoch 5/5  469/469 [====] - loss: 0.xx - accuracy: 0.9x
학습 시간 xx.xx초

평가 시작
313/313 [====] - loss: 0.xx - accuracy: 0.9x
정확도 0.9x
평가 시간 xx.xx초
```

---

## 전체 요약표

| Step | 이름 | 입력 모양 | 출력 모양 | 핵심 개념 |
|---|---|---|---|---|
| 1 | 데이터 준비 | (28,28) | (28,28,1) | 정규화, 채널 추가 |
| 2 | 패치화 | (B,28,28,1) | (B,49,16) | 이미지를 조각냄 |
| 3 | 패치 임베딩 | (B,49,16) | (B,49,8) | 의미 있는 벡터로 변환 |
| 4 | CLS 토큰 | (B,49,8) | (B,50,8) | 대표 토큰 추가 |
| 5 | 위치 임베딩 | (B,50,8) | (B,50,8) | 위치 정보 주입 |
| 6 | MHSA | (B,50,8) | (B,50,8) | 토큰 간 관계 계산 |
| 7 | Encoder 블록(부품) | (B,50,8) | (B,50,8) | 정규화+어텐션+MLP |
| 8 | 모델 조립 (Encoder ×6) | (B,28,28,1) | (B,50,8) | Step2~7을 연결 |
| 9 | Global Average Pooling | (B,50,8) | (B,8) | 50개 토큰 → 1개로 압축 |
| 10 | 분류 헤드 (Dense+Softmax) | (B,8) | (B,10) | 클래스 확률 계산 |
| 11 | 학습 및 평가 | - | - | 손실 계산, 정확도 측정 |
