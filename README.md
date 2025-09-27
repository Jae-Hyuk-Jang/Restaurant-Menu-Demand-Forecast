# Restaurant-Menu-Demand-Forecast
LG Aimesr 7th online hackathon

# Restaurant-Menu-Demand-Forecast
LG Aimesr 7th online hackathon


## 0) TL;DR

- **문제**: F&B 업장×메뉴의 **7일 매출 수량(h=1..7)** 예측
- **데이터 관점 핵심**: “미래 달력 특성(요일·주차 등)”을 **horizon에 맞춰 주입** / “제로(0) 제외 sMAPE”로 리더보드 정합
- **모델 구성(채널)**:
    - Global XGB Ensemble, Local XGB(by store), CatBoost(Poisson), LightGBM Horizon(+Quantile), sNaive(요일별 중앙값), Pos-only XGB, **Global LSTM**
- **검증**: cut_date(마지막 56일)에서 **14일 단위 롤링(bank) × 3개 오프셋(0/14/28일)** → 리더보드형 sMAPE(업장가중, A=0 제외) 최적화
- **메타 블렌딩**: 제약(심플렉스 + sNaive 상한) 하에서 **[Yg,Yl,Yc,Yh,Yr,Yn]** 가중 학습 + 메뉴별(Per-menu) 미세조정(+글로벌 수축)
- **후처리**:
    - ε-floor(업장×양수 5% 분위), **positive-day classifier** 확률 게이팅, **Soft Cap(최근 양수 퍼센타일 기반)**, Quantiles(Q10/Q90), **업장×horizon 보정**
- **오프라인 로그(참고)**
    - 채널 성능(window sMAPE): XGB Ens 0.659 / Cat(Poisson) 0.567 / LGB horizon 0.743 / PosReg 0.544 / LSTM 0.758 / **PosReg+LSTM 0.560**
    - 전역 제약 가중치(최종): **[Yg=0.079, Yl=0.047, Yc=0.108, Yh=0.532, Yr=0.124, Yn=0.111]** → **LGB Horizon**이 주도, sNaive·PosReg 보완
    - LB-like sMAPE(멀티-뱅크 평균): **0.3646**

---

## 1) 데이터 & 타깃

- **입력**: `train.csv`, `TEST_*.csv(10개)`, `sample_submission.csv`
- **타깃**: 일자별 `매출수량` (0 이상, 간헐적 수요 존재)
- **윈도우링**: **LOOKBACK=28**일 → **PREDICT=7**일 (h=1…7)
- **검증기간**: 마지막 56일(`VALID_DAYS=56`)을 리더보드 유사 평가로 사용

> 리더보드형 sMAPE:
> 
> 
> (1) **A=0 제외** (간헐적 수요의 노이즈 제거), (2) 업장 평균 후 **업장 가중** {담하:1.9, 미라시아:1.6} 반영
> 

---

## 2) 전처리 & 정규화

- 인코딩 자동 감지(`utf-8-sig`/`cp949`/`euc-kr`/`utf-8`), 이름 정규화(NFKC, 공백·NBSP 제거)
- `매출수량` 음수 클리핑(0 하한), 일자 파싱
- 키 정규화 함수로 **테스트 → 샘플 제출 컬럼 이름** 매핑 견고화

---

## 3) 특징(Feature) 설계

### 3-1. 캘린더 & 주기성

- `dow(요일)`, `month`, `is_weekend`, `is_holiday(고정양력)`, `is_big_holiday_window(명절 주변)`,
- 주기 성분: `dow_sin`, `dow_cos`, `woy(주차)`, `woy_sin`, `woy_cos`
- 요일 one-hot: `dow_0`…`dow_6`

### 3-2. 시계열 집계/지연

- 지연: `lag_1`, `lag_7`, `lag_14`
- 이동평균: `roll_mean_7`, `roll_mean_14`
- 차분·증감: `diff_1(클리핑)`, `pct_chg_1(클리핑)`
- 누적/합계/평균: `sum_7`, `sum_14`, `mean_14`, `sum_prev7(이전 주 동일 요일 합)`
- 제로/양수 패턴: `zero_share_7`, `zero_share_28`, `nonzero_gap(마지막 양수 이후 경과)`, `zero_rate7`
- 상호작용: `wknd_x_mean7`, `holi_x_mean14`
- 중앙값: `roll_med_7`

> 최종 사용 특성 수: 36개 (로그: n_features: 36)
> 

**표** 

| 분류 | 컬럼 | 설명/의도 |
| --- | --- | --- |
| 달력 | `dow`, `month`, `is_weekend`, `is_holiday`, `is_big_holiday_window` | 수요 주기/휴일 효과 |
| 주기 | `dow_sin`, `dow_cos`, `woy`, `woy_sin`, `woy_cos` | 부드러운 주기 표현 |
| 요일 one-hot | `dow_0`~`dow_6` | 모형의 비선형 포착 보강 |
| 지연/평균 | `lag_1`,`lag_7`,`lag_14`,`roll_mean_7`,`roll_mean_14` | 최근 레벨/주기상태 |
| 변화 | `diff_1`,`pct_chg_1` | 급격한 변동 포착 |
| 누적/합 | `sum_7`,`sum_14`,`sum_prev7`,`mean_14` | 최근량·시즌성 |
| 제로/양수 | `zero_share_7`,`zero_share_28`,`nonzero_gap`,`zero_rate7` | 간헐성/재고성 |
| 상호작용 | `wknd_x_mean7`,`holi_x_mean14` | 요인 결합 효과 |
| 통계 | `roll_med_7` | 극값 억제, 강건성 |

---

## 4) 윈도우링 & 검증 스킴

- 모든 메뉴에 대해 **길이 28**의 입력 창과 **7일 타깃** 슬라이딩
- 검증은 **56일 구간**을 **7일 단위**로 굴리며 **오프셋(0/14/28)** 3개 **Validation bank** 구성
- 각 bank 내에서 리더보드형 sMAPE를 계산 → **가중 최적화/메타블렌딩**에 사용

<img width="1090" height="1368" alt="Image" src="https://github.com/user-attachments/assets/b07d6689-1f66-4347-b913-6c714026455c" />

---

## 5) 베이스 모델(채널) 설계

> 채널 표기: Yg(Global XGB), Yl(Local XGB per-store), Yc(CatBoost Poisson), Yh(LGB Horizon), Yr(Positive-only XGB), Yn(sNaive)
> 
- **Global XGB Ensemble**: seed `{41,42,777,1337,2024,31415}` 앙상블, Tweedie(1.3), hist, GPU
- **Local XGB**: 업장 단위로 개별 다중-아웃풋; 데이터가 충분한 업장만 학습
- **CatBoost (Poisson)**: count-like 타깃에 강건, GPU, 조기중단 사용
- **LightGBM Horizon**: **horizon을 명시적 피처화**하고 **미래 달력**을 **h**만큼 밀어 주입
- **Quantile LGB (Q10/Q50/Q90)**: 후처리 floor/cap/soft-cap 등에 사용
- **sNaive**: 최근 4주 동일 요일 **중앙값**
- **Positive-only XGB**: **양수만 log-회귀** → 간헐수요 안정화
- **Global LSTM**: (B,T,F) 입력, last hidden → 7-아웃; **PosReg와 α-블렌딩** (α=0.45)

**채널별 검증(window sMAPE, 예시 로그)**

- XGB Ens **0.6591** / Local XGB(개수=9) / Cat(Poisson) **0.5667** / LGB Horizon **0.7433**
- PosReg **0.5435** / LSTM **0.7582** / **PosReg+LSTM 0.5603**

<img width="1208" height="1294" alt="Image" src="https://github.com/user-attachments/assets/42ccd9c5-5eea-47da-8b2b-bf7a2e162182" />

---

## 6) Positive-day classifier & 게이팅

- **목표**: “그날이 **양수일** 것인가?” 확률 php_h 예측 → 후처리에서 **부드럽게 게이팅**
- **모형**: GradientBoostingClassifier + **Isotonic Calibration**
- **입력**: 마지막 시점 특성 + 최근 7/14일 간단 통계
- **스쿼시**: p′=(1−a)p+a⋅0.5p'=(1-a)p + a·0.5 (a=0.035) → 과신 억제
- **게이팅**: 예측 y^h←y^h⋅(0.9+0.1⋅ph′)\hat{y}_h \leftarrow \hat{y}_h \cdot (0.9 + 0.1·p'_h)

---

## 7) 후처리(Post-process)

1. **ε-floor**: 업장별 양수 5% 분위(최소 EPS_MIN_FLOOR) → **Q10**와 결합해 바닥 보장
2. **Soft Cap**: 최근 양수값의 p90×1.12를 기본 캡으로, **Q90×1.10**과 최소값 적용
3. **업장×Horizon 보정**: bank 기반 **중앙값 비율**로 **±12%** 윈저라이징
4. **All-zero Guard**: 극단적 상황에서 `max(sNaive, floor)` 대체

---

## 8) 메타 블렌딩(제약 & 메뉴별 미세조정)

### 8-1. 전역 제약 가중(6채널)

- 가중 w₆ = [Yg,Yl,Yc,Yh,Yr,Yn]
- **제약**: 심플렉스(합=1), **sNaive 상한**(GLOBAL_SNAIVE_CAP=0.30)
- **목표**: bank별 리더보드형 sMAPE 근사 기울기(안정형 MSE도 혼합 가능)로 GD → **프로젝션**

**결과(예시 로그)**:

**W_GLOBAL6 = [0.079, 0.047, 0.108, 0.532, 0.124, 0.111]**

→ **LGB Horizon(Yh)**이 주도, **PosReg(Yr)**·**sNaive(Yn)**이 보조, **Cat(Yc)**가 서포트

### 8-2. 메뉴별(Per-menu) 가중 & 수축

- bank0에서 **거친 그리드(0.1 step)** → **좌표하강(0.02)**
- 메뉴별 가중 w^\hat{w}을 전역 ww와 **수축(shrinkage)**: 희소도에 따라 수축 강도↑
- 최종 ww는 다시 심플렉스/상한으로 **프로젝션**

<img width="1982" height="1038" alt="Image" src="https://github.com/user-attachments/assets/c690c1d3-152a-4556-b319-6691af1dac13" />

---

## 9) 추론(테스트) 파이프라인

1. 테스트 파일별로 동일한 **특징 생성**
2. 채널별 예측 + **Yr' (PosReg+LSTM)**
3. **메뉴별 가중**(전역 수축 반영)으로 합성
4. **후처리**(ε-floor, Q10/Q90, Soft Cap, 게이팅, 업장×h bias)
5. `sample_submission.csv` 스키마로 매핑/저장

<img width="898" height="1230" alt="Image" src="https://github.com/user-attachments/assets/308423fd-d294-476c-ae50-2967007920b2" />

---

## 10) 오프라인 결과 & 해석 (노트북 로그 기반)

- **채널 단품** sMAPE는 **LGB Horizon, LSTM**이 상대적으로 높았지만,
    - Horizon 모델은 “미래 달력 일치” 장점 때문에 **가중 합성에서 주도**
    - **PosReg+LSTM 블렌드**는 간헐성 구간에서 **스무딩 효과** 제공
- **전역 가중**에서 **Yh 0.53**으로 가장 큰 비중 → 시즌성/달력 민감도가 강함을 시사
- **LB-like sMAPE=0.3646** (우리의 오프라인 척도이며, 실제 LB와 절대값은 다를 수 있음)

---

## 11) 교훈/회고

- **정합 지표**를 맞추는 설계(제로 제외 sMAPE, 업장 가중)가 대회 성능에 중요
- **미래 달력 주입 Horizon 모델** + **확률 게이팅** + **Soft Cap**이 간헐 수요에서 안정성 제공
- **Per-menu 가중 + 전역 수축**은 과적합 완화에 유효
- *딥러닝 단품(LSTM)**은 단독 성능이 낮더라도 **블렌드要**로 역할

---


## 12) 수식 & 의사코드

**리더보드형 sMAPE(아이템 i, horizon t, A>0만)**

$${sMAPE}_i=\frac{1}{T_i}\sum_{t: A_{it}>0}\frac{2\lvert A_{it}-P_{it}\rvert}{\lvert A_{it}\rvert+\lvert P_{it}\rvert+\epsilon}
\quad\Rightarrow\quad
\text{LB}=\frac{\sum_{s} w_s \cdot \text{avg}_i(\text{sMAPE}_i\ \text{in store }s)}{\sum_s w_s}$$

**메타 블렌딩(제약) 의사코드**

```
Given banks B, channels Yg,Yl,Yc,Yh,Yr,Yn and weights w (sum=1, 0≤w≤1, w[Yn]≤cap)
repeat:
  grad ← 0
  for each bank b in B:
    for each row r in b:
      A ← r.A; mask ← (A>0)
      P ← postprocess(w · [Y*], r.quantiles, r.P, eps, cap)
      C ← stack([Y*])[mask]
      grad += (-2·(A-P) @ C) * store_weight[r.store]
  w ← project_simplex_with_cap(w - lr·grad/num)
until converge

```

**후처리 요약**

- `floor = max(eps_store, Q10)` (또는 `max(eps, Q10, 0.5·Q50)`)
- `P ← max(P, floor)`
- `P ← P · (0.9 + 0.1·p')`, `p'=(1-a)p + a·0.5`
- `P ← min(P, min(SoftCap, 1.10·Q90))`
- `P_h ← P_h · bias(store,h)`

---

## 13) 다이어그램 

### (1) 전체 파이프라인

<img width="1454" height="1414" alt="Image" src="https://github.com/user-attachments/assets/415d652b-ded1-45e0-ab96-a4fa7047d7fe" />

### (2) Validation bank 타임라인

<img width="2836" height="491" alt="Image" src="https://github.com/user-attachments/assets/8eeb63dd-6001-4e3c-b960-a4f42d5c71a9" />

### (3) 후처리 파이프라인

<img width="2382" height="504" alt="Image" src="https://github.com/user-attachments/assets/936ac822-8730-4a92-95be-50bb5d9783c9" />

---
