# 비젼 트랜스포머로 손글씨를 인식하기 (수정본)
#
# ------------------------------------------------------------
# [수정 이력]
# 1. (심각) x_test 전처리 누락 수정
#    -> 기존: x_train만 스케일링(÷255)과 채널 차원 추가를 했고 x_test는 안 함
#    -> evaluate() 실행 시 모양 불일치(shape mismatch) 에러가 남
#    -> x_train과 동일하게 x_test도 전처리하도록 수정
#
# 2. (설계 버그) encoder_block 안에서 LayerNormalization 레이어를 재사용하던 문제 수정
#    -> 기존: norm 이라는 레이어 객체 하나를 어텐션 앞, MLP 앞에서 두 번 호출
#       Keras는 레이어 객체를 재사용하면 같은 학습 파라미터(gamma, beta)를 공유함
#       -> 표준 Transformer 구조와 달라지고 표현력이 줄어듦
#    -> norm1(어텐션용), norm2(MLP용) 두 개의 독립된 레이어로 분리
#
# 3. (사소) 주석과 실제 값 불일치 수정
#    -> "40 차원" 이라는 주석이 실제 계산값(MY_MLP_MULT * MY_HIDDEN = 80)과 다름
#    -> 실제 계산되는 값을 출력해서 확인할 수 있도록 print 추가
#
# 4. (설계 노트) CLS 토큰 생성 방식을 표준 ViT 방식으로 교체
#    -> 기존: 패치들의 평균(GAP)을 Dense로 변환해서 CLS 토큰으로 사용
#       (입력값에 종속적인 방식, 원 논문과 다름)
#    -> 수정: 입력과 무관하게 독립적으로 학습되는 파라미터 벡터를
#       배치 크기만큼 복제(tile)해서 사용하는 표준 ViT 방식으로 변경
# ------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from time import time


# ============================================================
# 0단계. 문제 정의
# ------------------------------------------------------------
# - 28x28 흑백 손글씨 숫자(MNIST) 이미지를 입력받아
#   0~9 중 어떤 숫자인지 분류하는 Vision Transformer를 만든다.
# - 이미지를 패치(작은 조각)들로 나눈 뒤, 각 패치를 하나의 "단어"처럼
#   취급해서 문장을 이해하듯 Transformer로 이미지를 이해하게 한다.
# ============================================================


# 하이퍼 파라미터 지정
MY_SHAPE = (28, 28, 1)             # 손글씨 이미지 데이터 모양
MY_EPOCH = 5                       # 반복 학습 수
MY_BATCH = 128                     # 배치 수
MY_LEARN = 0.005                   # 학습율
MY_CLASS = 10                      # 분류화 대상의 수

MY_PATCH = 7                       # 한 면의 패치 수 (7x7 = 49개 패치)
MY_TOKEN = 50                      # 총 입력 토큰 수 (49개 패치 + CLS 토큰 1개)
MY_ENCODER = 6                     # 총 인코더 수
MY_MLP_MULT = 10                   # MLP 확장 배수 (n_hidden에 곱해지는 배수)

MY_HIDDEN = 8                      # 패치 임베딩 차원 수
MY_HEAD = 2                        # 어텐션 동시 계산 머리 수


# ============================================================
# 1단계. 데이터 준비 (분석 -> 설계 -> TDD 순서로 진행)
# ------------------------------------------------------------
# [TDD] 기대하는 결과:
#   - x_train, x_test 모두 0~1 사이 실수(float32)
#   - x_train, x_test 모두 모양이 (28, 28, 1) 이어야 함 (채널 차원 포함)
# ============================================================

mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

# [수정 1] x_train과 x_test를 "동일한 방식"으로 함께 전처리한다.
# 스케일링(÷255)으로 0~1 사이 값으로 만들고,
# [..., None]으로 채널 차원(흑백=1)을 추가해서 모델 입력 모양과 맞춘다.
x_train = (x_train.astype('float32') / 255.0)[..., None]
x_test = (x_test.astype('float32') / 255.0)[..., None]

# 검증: 두 데이터의 모양이 같은 규칙으로 처리되었는지 확인
print('[검증] x_train 모양:', x_train.shape, ' 값 범위:', x_train.min(), '~', x_train.max())
print('[검증] x_test 모양 :', x_test.shape, ' 값 범위:', x_test.min(), '~', x_test.max())
assert x_train.shape[1:] == (28, 28, 1)
assert x_test.shape[1:] == (28, 28, 1)   # 수정 전에는 (28, 28)이라 여기서 에러가 났을 것

# 학습용 데이터 배치 처리
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(MY_BATCH)
)

# 평가용 데이터 배치 처리
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .shuffle(10000)
    .batch(MY_BATCH)
)


# ============================================================
# 2단계. 이미지 패치 처리 [흐름도 ①, Step 2]
# ------------------------------------------------------------
# [문제 정의] 이미지를 작은 정사각형 조각(패치)들로 잘라야 한다.
# [설계] 28x28 이미지를 4x4 크기 조각 49개(7x7)로 나눈다.
# [TDD] 기대하는 결과:
#   입력: (128, 28, 28, 1)
#   출력: (128, 49, 16)   <- 49개 패치, 각 패치는 4x4=16개 픽셀값
# ============================================================

def PatchLayer():
    size = 28 // MY_PATCH      # 패치 한 변의 크기 (28 // 7 = 4)
    num = MY_PATCH * MY_PATCH  # 총 패치 개수 (7 * 7 = 49)
    dim = size * size          # 패치 하나의 픽셀 수 (4 * 4 = 16)

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

    layer = layers.Lambda(extract)
    return layer


# 테스트용 코드 (TDD: 위에서 예상한 모양과 실제 결과가 같은지 확인)
x = tf.random.normal([MY_BATCH, 28, 28, 1])
patchify = PatchLayer()
tmp = patchify(x)
print('\n[PatchLayer 테스트]')
print('입력', x.shape)
print('출력', tmp.shape)
assert tmp.shape == (MY_BATCH, 49, 16)


# ============================================================
# 3단계. Multi-Head Attention 구현 [흐름도 ⑤의 부품, Step 6]
# ------------------------------------------------------------
# [문제 정의] 각 패치(토큰)가 다른 패치들과 얼마나 관련 있는지
#             계산해서 정보를 섞어주는 장치가 필요하다.
# [설계] Q(질문), K(열쇠), V(값) 벡터를 만들고,
#        Q와 K를 비교해 "얼마나 주목할지" 점수를 구한 뒤(Attention),
#        그 점수로 V를 가중합해서 새로운 표현을 만든다.
#        이 과정을 여러 "머리(head)"로 나눠 동시에 수행한다.
# [TDD] 기대하는 결과:
#   입력: (128, 50, 8)
#   출력: (128, 50, 8)   <- 모양은 그대로, 내용만 문맥을 반영해 바뀜
# ============================================================

class MyMHA(layers.Layer):
    def __init__(self, n_hidden, n_head):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.d_head = n_hidden // n_head    # 각 머리가 계산하는 차원 수 (= dk)

        # Q, K, V 벡터 전환용 행렬
        self.wq = layers.Dense(n_hidden, use_bias=True)
        self.wk = layers.Dense(n_hidden, use_bias=True)
        self.wv = layers.Dense(n_hidden, use_bias=True)

        # 최종 결과
        self.wz = layers.Dense(n_hidden, use_bias=True)

    def call(self, x, training=False):
        b = tf.shape(x)[0]                # 한 배치 크기, 128
        t = tf.shape(x)[1]                # 50

        # q, k, v 벡터 전환
        q = self.wq(x)                    # 데이터 모양: (128, 50, 8)
        k = self.wk(x)
        v = self.wv(x)

        # 머리로 나누기: (128, 50, 8) -> (128, 50, 2, 4)
        q = tf.reshape(q, [b, t, self.n_head, self.d_head])
        k = tf.reshape(k, [b, t, self.n_head, self.d_head])
        v = tf.reshape(v, [b, t, self.n_head, self.d_head])

        # Attention 계산 전에 모양 준비: (128, 50, 2, 4) -> (128, 2, 50, 4)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Attention 계산
        scale = tf.cast(self.d_head, tf.float32) ** -0.5      # 루트 dk로 나눔
        score = tf.matmul(q, k, transpose_b=True) * scale     # 모양: (128, 2, 50, 50)
        att = tf.nn.softmax(score, axis=-1)                   # 합이 1이 되도록 정규화
        final = tf.matmul(att, v)                             # 모양: (128, 2, 50, 4)

        # 머리 합치기
        final = tf.transpose(final, [0, 2, 1, 3])             # (128, 50, 2, 4)
        merge = tf.reshape(final, [b, t, self.n_hidden])      # (128, 50, 8)
        z = self.wz(merge)                                    # 최종 z 결과

        return z


# 테스트용 코드
x = tf.random.normal([MY_BATCH, MY_TOKEN, MY_HIDDEN])
mha = MyMHA(MY_HIDDEN, MY_HEAD)
y = mha(x)
print('\n[MyMHA 테스트]')
print('입력', x.shape)
print('출력', y.shape)
assert y.shape == (MY_BATCH, MY_TOKEN, MY_HIDDEN)


# ============================================================
# 4단계. 인코더 블록 만들기 [흐름도 ⑤의 부품, Step 7]
# ------------------------------------------------------------
# [문제 정의] Attention 하나만으로는 표현력이 부족하다.
#             정규화 + Attention + 잔차연결(skip connection) +
#             정규화 + MLP + 잔차연결을 하나의 블록으로 묶어야 한다.
# [설계] Pre-Norm 구조: "정규화 -> 연산 -> 원본과 더하기" 순서를 사용한다.
#        어텐션용 정규화와 MLP용 정규화는 서로 다른 통계(평균/분산)를
#        학습해야 하므로 반드시 "서로 다른" 레이어여야 한다.
# [TDD] 기대하는 결과:
#   입력: (128, 50, 8)
#   출력: (128, 50, 8)
# ============================================================

def encoder_block(x, n_hidden, n_head, mlp_mult):
    # [수정 2] 어텐션용(norm1)과 MLP용(norm2) 정규화 레이어를
    # 각각 독립적으로 만든다. 하나의 레이어를 두 번 재사용하면
    # 두 지점이 같은 gamma/beta 파라미터를 공유하게 되어
    # 서로 다른 통계를 학습할 수 없게 된다.
    norm1 = layers.LayerNormalization(epsilon=1e-5)
    norm2 = layers.LayerNormalization(epsilon=1e-5)

    # 어텐션 서브블록: 정규화 -> MHA -> 잔차연결
    y = norm1(x)
    y = MyMHA(n_hidden, n_head)(y)
    y = layers.Add()([x, y])

    # MLP 서브블록: 정규화 -> MLP -> 잔차연결
    w = norm2(y)

    # [수정 3] 주석이 실제 값과 다르던 부분: 실제 계산값을 출력해서 확인
    expand = mlp_mult * n_hidden         # MY_MLP_MULT(10) * MY_HIDDEN(8) = 80 차원
    y = layers.Dense(expand, activation=tf.nn.gelu)(w)
    y = layers.Dense(n_hidden)(y)        # 다시 n_hidden(8) 차원으로 축소
    y = layers.Add()([w, y])

    return y


# 테스트용 코드
x = tf.random.normal([MY_BATCH, MY_TOKEN, MY_HIDDEN])
y = encoder_block(x, MY_HIDDEN, MY_HEAD, MY_MLP_MULT)
print('\n[encoder_block 테스트]')
print('MLP 확장 차원 실제값:', MY_MLP_MULT * MY_HIDDEN, '(주석 오류 수정 확인용)')
print('입력', x.shape)
print('출력', y.shape)
assert y.shape == (MY_BATCH, MY_TOKEN, MY_HIDDEN)


# ============================================================
# 5단계. 위치 임베딩 [흐름도 ④, Step 5]
# ------------------------------------------------------------
# [문제 정의] Attention은 순서를 모른다 (모든 토큰을 동시에 봄).
#             패치가 이미지의 "어느 위치"에 있었는지 알려줘야 한다.
# [설계] sin/cos 함수로 각 위치마다 고유한 패턴을 만들어 더해준다.
# ============================================================

def pos_embed(n_token, d_hidden):
    pe = np.zeros((n_token, d_hidden), dtype=np.float32)
    for i in range(n_token):
        for j in range(d_hidden):
            if j % 2 == 0:
                pe[i, j] = np.sin(i / 10000 ** (j / d_hidden))
            else:
                pe[i, j] = np.cos(i / 10000 ** ((j - 1) / d_hidden))
    pe = tf.constant(pe, dtype=tf.float32)
    return pe


# 테스트용 코드
pe = pos_embed(MY_TOKEN, MY_HIDDEN)
print('\n[pos_embed 테스트]')
print(pe.shape)
print('위치 0 정보', pe[0].numpy())
print('위치 1 정보', pe[1].numpy())
print('위치 49 정보', pe[49].numpy())
assert pe.shape == (MY_TOKEN, MY_HIDDEN)


# ============================================================
# 6단계. CLS(클래스) 토큰 레이어 [흐름도 ③, Step 4]
# ------------------------------------------------------------
# [문제 정의] 여러 패치 토큰들의 정보를 최종적으로 모아서
#             "이 이미지는 무슨 숫자다"라고 판단할 대표 토큰이 필요하다.
# [설계 - 수정 4] 표준 ViT 방식: 입력 이미지 내용과 무관하게
#             독립적으로 학습되는 벡터 하나를 준비해두고,
#             배치 크기만큼 복제(tile)해서 각 이미지 앞에 붙인다.
#             (기존처럼 "패치 평균"을 변환해서 쓰지 않음)
# [TDD] 기대하는 결과:
#   입력 패치 토큰: (128, 49, 8)
#   출력: (128, 1, 8)  <- 이미지 내용과 무관하게 같은 초기값에서 시작
# ============================================================

class ClsToken(layers.Layer):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden

    def build(self, input_shape):
        # 입력과 무관한, 독립적으로 학습되는 파라미터 벡터를 하나 만든다.
        self.cls = self.add_weight(
            shape=(1, 1, self.n_hidden),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, x):
        b = tf.shape(x)[0]                     # 배치 크기
        return tf.tile(self.cls, [b, 1, 1])    # 배치 크기만큼 복제


# 테스트용 코드
x = tf.random.normal([MY_BATCH, 49, MY_HIDDEN])
cls_layer = ClsToken(MY_HIDDEN)
cls_out = cls_layer(x)
print('\n[ClsToken 테스트]')
print('입력(패치 토큰)', x.shape)
print('출력(CLS 토큰)', cls_out.shape)
assert cls_out.shape == (MY_BATCH, 1, MY_HIDDEN)


# ============================================================
# 7단계. 비젼 트랜스포머 전체 조립
# ------------------------------------------------------------
# 아래 각 줄의 주석에 붙은 ①~⑦은 강의자료(md 문서) 상단의
# "전체 그림 먼저 보기" 흐름도 번호와 1:1로 대응된다.
# 입력 이미지 -> ①패치화 -> ②패치 임베딩 -> ③CLS 토큰 추가
# -> ④위치 임베딩 추가 -> ⑤인코더 반복 -> ⑥GAP -> ⑦분류(Dense+Softmax)
# ============================================================

# 입력층
input = keras.Input(shape=MY_SHAPE)

# ① 패치로 자르기 [흐름도 ①, Step 2]
patches = PatchLayer()(input)
print('\n[모델 조립 과정]')
print('패치화 후', patches.shape)

# ② Linear 임베딩: 16차원 -> 8차원 [흐름도 ②, Step 3]
tokens = layers.Dense(MY_HIDDEN)(patches)
print('임베딩 후', tokens.shape)

# [수정 4] CLS 토큰: 표준 ViT 방식으로 생성 (입력 무관, 독립 학습 파라미터)
ctoken = ClsToken(MY_HIDDEN)(tokens)

# ③ CLS 토큰 추가: 49개 패치 토큰 맨 앞에 붙이기 -> 총 50개 토큰 [흐름도 ③, Step 4]
concat = layers.Concatenate(axis=1)
tokens = concat([ctoken, tokens])
print('클래스 토큰 추가 후', tokens.shape)

# ④ 위치 임베딩 더하기 [흐름도 ④, Step 5]
pe = pos_embed(MY_TOKEN, MY_HIDDEN)
x = tokens + pe
print('위치 임베딩 후', x.shape)

# ⑤ Transformer Encoder x 6회 반복 [흐름도 ⑤, Step 6·7·8]
#    (Step 6: MHSA 부품, Step 7: 정규화+MHSA+MLP를 묶은 encoder_block,
#     Step 8: 그 encoder_block을 MY_ENCODER번 반복해서 쌓는 부분이 바로 여기)
for _ in range(MY_ENCODER):
    x = encoder_block(x, MY_HIDDEN, MY_HEAD, MY_MLP_MULT)
print('인코더 추가 후', x.shape)

# ⑥ Global Average Pooling: 50개 토큰 -> 1개 대표 벡터로 압축 [흐름도 ⑥, Step 9]
x = layers.GlobalAveragePooling1D()(x)
print('GAP 처리 후', x.shape)

# ⑦ Dense(10) + Softmax: 분류 헤드, 1개 벡터 -> 10개 클래스 확률 [흐름도 ⑦, Step 10]
output = layers.Dense(MY_CLASS, activation='softmax')(x)
print('최종 출력', output.shape)

# 모델 생성
model = keras.Model(input, output)
model.summary()


# ============================================================
# 8단계. 학습 및 평가 [Step 11]
# ============================================================

adam = keras.optimizers.Adam(learning_rate=MY_LEARN)
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('\n학습 시작')
begin = time()
model.fit(train_ds, epochs=MY_EPOCH)
print('학습 시간', time() - begin)

print('\n평가 시작')
loss, acc = model.evaluate(test_ds)   # 수정 1 덕분에 이제 에러 없이 끝까지 실행됨
print('정확도', acc)
print('평가 시간', time() - begin)
