# SCLM 설계 — DeepSeek V4.1 Flash 로컬 고속 추론

표기: **[확인]** config/weight index/로컬 측정으로 확인 · **[추정]** 그 수치로 계산 · **[논문]** 발표된 기법 · **[아이디어]** 미발표 가설

## 1. 목표와 제약

- 목표: 단일 사용자 decode **20~60 tok/s**, 품질은 FP4 원본 대비 최소 손실.
- 제약: **클라우드 사용 금지**, **Windows 11에서 완결**(네이티브 + WSL2 허용).
- 하드웨어 [확인, 2026-11-08 측정, `bench/`]

| 자원 | 사양 | 실측 |
|---|---|---|
| GPU | RTX 5080 16GB, PCIe 5.0 x16, NVFP4 | VRAM 복사 758GB/s(읽기+쓰기 합산), 화면 연결 상태 가용 14.6GiB |
| PCIe | pinned | H2D/D2H 56GB/s (1MiB 이상), 64KiB는 10~12GB/s, pageable 16.7GB/s |
| 동기화 | WDDM | launch+sync 9.4µs, 40KiB 왕복+sync 14.4µs, **mapped 메모리 ping-pong 1.1µs** |
| RAM | 96GB (16+32 혼합 4장) DDR5-4200 듀얼채널 | CPU 읽기 **50~54GB/s** (1스레드도 50, CCD 연결 상한), 쓰기 28GB/s |
| RAM 동시 | CPU 읽기 + H2D DMA | **합계 58~59GB/s** (CPU 27~28 + DMA 31) |
| CPU | Ryzen 7 9800X3D 8C/16T, AVX-512 | L3 읽기 743GB/s (8스레드) |
| SSD | Solidigm P44 Pro 2TB (C:, 여유 753GB) | 순차 7.4GB/s, 4KiB 랜덤 QD1 1.45만 / QD32 53만 IOPS |
| pinned 한도 | Windows | 48GiB 할당 성공 (탐색 상한까지) |
| 기타 | H:는 `C:\devdrive.vhdx`(ReFS Dev Drive, C:와 공간 공유), D: HDD, WSL2 Ubuntu 24.04, CUDA 13.4, VS 18 | |

**전체 병목은 RAM 대역폭이다.** CPU 코어만으로는 50GB/s에서 막히지만 DMA는 IO die에서 DRAM을 직접 읽어 56GB/s가 나오고, 둘을 섞으면 합계 58~59GB/s까지 올라간다. 따라서 R7 스케줄러는 CPU 계산과 GPU 승격을 섞어서 DRAM을 채우는 것을 목표로 한다. 측정 당시 백그라운드 앱이 RAM 약 29GB를 점유하고 있었으므로 실행 시에는 정리가 필요하다.

## 2. 모델 사실

- [확인] hidden 5120, 40층, routed expert 384(top-6) + shared 1, expert 중간 차원 2304.
- [확인] 원본 dtype: expert FP4, 나머지 FP8 (block 32×32, ue8m0 scale). 체크포인트 총 510GB.
- [확인] scoring `sqrtsoftplus`, `noaux_tc`, routed_scaling 1.5, mHC(`hc_mult` 4), sliding window 128, 압축·희소 attention(`compress_ratios`, indexer).
- [확인] KV 원천 층 `[2, 8, 14, 20]`, Engram 층 `[1, 14]`, MTP 3층(DSpark: block 5, 128 experts top-3, markov head).
- [확인] expert 1개 35.4M(FP4 e2m1, K 방향 32개당 e8m0 scale 1개, 약 18.8MB) → routed 총 543.6B. decode 토큰당 routed 8.5B.
- [확인, 공식 `inference/model.py`] 상시 경로는 층당 attention 126.6M(wq_a·wq_b·wkv·wo_b는 FP8, wo_a는 bf16 block-diagonal 33.5M), shared expert FP8 35.4M, gate bf16, mHC fp32 약 2M이다. 여기에 Engram wkv(6144→25600, FP8, 157M, 2개 층)와 head(bf16, 0.66B)를 더하면 **상시 약 7.8B**, decode 총 약 16.3B다.
- [확인] **"인코더-디코더" 분리는 코드에 없다.** 40층 모두 prefill과 decode에서 실행된다. 이전의 "prefill은 앞 20층만" 추정은 틀렸다.
- [확인] attention: 모든 층이 sliding window 128을 보고, 2~39층은 압축 KV(2~19층 2:1, 20~39층 1:1)를 추가로 본다. 압축 KV는 원천 층(2, 8, 14, 20)이 만들고 뒤 층들이 공유한다. indexer(2, 8, 14, 20, 24, 28, 32, 36층)가 압축 위치 중 top-512를 고르고, 20층의 candidate 블록(2048×8)이 21층 이후의 1차 필터가 된다. 토큰당 attention 비용은 문맥 길이와 거의 무관하다(indexer 점수 계산만 선형).
- [확인] 수치: FP8/FP4 GEMM 앞에서 활성값을 FP8(32개 블록, 2의 거듭제곱 scale)로 fake-quant한다. window KV는 FP8, 압축 KV는 FP4(16개당 e4m3 scale), indexer q/k는 FP4다. **학습부터 FP8 활성값을 썼으므로 CPU 커널의 8bit 활성 양자화는 품질 비용이 없다.**
- [확인] 공식 코드는 tilelang 커널과 8-GPU 분산(`torchrun`, MP=8)을 전제로 한다. 다중 토큰 decode(speculative 검증)는 구현되어 있지 않다.
- [확인] 샤드 구성: 층마다 7.4GB 샤드 1개(dense+routed, 총 296GB), Engram 2개(층 1, 14, 각 101.5GB, 총 203GB), MTP 7.9GB, vision 2.3GB, embed/head 각 1.3GB.
- [확인, `inference/engram.py`] Engram 테이블은 층마다 (2/3/4-gram) × 8 head = 24개의 소수 크기 구간(각 약 1,600만 행)으로 나뉜다. 토큰당 층마다 정확히 24행(FP8 256차원 + e8m0 scale 8개)을 읽는다. 해시는 정규화된 압축 어휘(99,092개) 기준 n-gram이라 토큰 id만으로 미리 계산된다.
- [추정] 토큰당 48행 랜덤 읽기 → QD32에서 0.1ms 수준. 2-gram 행은 자주 반복되어 RAM 캐시 적중이 높고, 4-gram 행은 대부분 차갑다.
- [확인] DSpark: MTP 3블록(window attention만, expert 128개 top-3). 입력은 37~39층 attention 입력의 hc 평균이다. 마지막 토큰 + noise 토큰 4개로 5칸 블록을 한 번에 draft하고, Markov head(rank 256)가 순차적으로 보정하며, confidence head가 신뢰도를 낸다.

## 3. 메모리 계층

```
GPU 16GB : 상시 경로(NVFP4/3bit) ~3.8GB · DSpark ~2.5GB · hot expert 캐시 ~6.5GB · KV/작업 ~1.5GB
RAM 96GB : 압축 routed expert 전체 (Plan C 약 50GB) · codebook/LUT는 L3 상주
SSD (C:) : 압축 Engram · student 파일 (FP4 원본은 Pass 1 중에만 샤드 단위로 존재, 8.1절)
```
- 화면 출력은 메인보드(내장 Radeon)에 연결해 5080 VRAM을 전부 확보한다.

## 4. 압축 (Plan C: 평균 약 0.9bpw, 사용 빈도 가중 정밀도)

| ID | 내용 | 근거 |
|---|---|---|
| C1 | 도메인 데이터 기반 router 가중 expert 가지치기 (384 → 256~288) | [논문] REAP |
| C2 | 층별 공유 기저 + expert 잔차 분해 | [논문] MoBE, D²-MoE |
| C3 | Hadamard 회전 + trellis/벡터 양자화 | [논문] QuIP#, QTIP, AQLM |
| C4 | 꼬리 expert 1bit 미만 (이진화 + 저랭크 + N:M) | [논문] QMoE, STBLLM, LittleBit |
| C6/Q1 | 사용 빈도 가중 비트 배분: hot 3~4bit(GPU), 꼬리 0.5~1bit | [논문] MC-MoE, HOBBIT |
| C7 | GPU 캐시 무손실 엔트로피 코딩 | [논문] QMoE, DFloat11 |
| Q2 | 중첩 bit-plane 저장 (저정밀 = 고정밀의 앞부분) | [논문] Matryoshka Quant, Any-Precision |
| Q3 | 불확실 토큰만 하위 plane까지 읽어 재계산 | [아이디어] |
| Q4 | 층별 공유 저랭크 오차 보정항을 GPU에 고정밀로 | [논문] EoRA, LQER |
| Q5 | 가지치기·캐시 고정·비트 배분을 사용 도메인에 맞춤 | — |
| C8 | **Engram 곱 양자화(PQ)**: 256차원 행을 16개 부분공간 × 8bit 코드로 바꿔 행당 16B. 203GB → 약 12~25GB | [논문] 임베딩 PQ 압축 응용 |
| C9 | **DSpark drafter 극단 압축**: drafter 품질은 채택률(속도)에만 영향을 주고 결과는 바뀌지 않으므로(무손실 검증) expert 가지치기 + 1~1.5bit로 VRAM 1.5~2.5GB | [아이디어] |

## 4.1 Phase 1 첫 실측: 레이어 10 routed expert 압축 (2026-11-09, `data/probe_L10_results.json`)

실제 FFN 입력 16,384토큰, 원본 FP4(4.25bpw) 대비. MoE NMSE는 실제 라우팅 가중치로 합친 출력(shared expert 포함) 기준.

| 방법 | bpw | 가중치 NMSE | MoE 출력 NMSE | 코사인 |
|---|---|---|---|---|
| RTN 3bit g64 | 3.25 | 3.4% | 4.3% | 0.979 |
| RTN 3bit g64 + Hadamard | 3.25 | 3.3% | **2.8%** | 0.980 |
| RTN 2bit g64 | 2.25 | 12.4% | 17.2% | 0.922 |
| RTN 2bit g64 + Hadamard | 2.25 | 11.4% | **12.5%** | 0.928 |
| VQ d8, 4,096 코드 (+Had) | 1.50 | 17.2% | 24.4% (22.3%) | 0.888 |
| VQ d8, 256 코드 (+Had) | 1.00 | 32.6% | 54% (51%) | 0.768 |
| VQ d16, 4,096 코드 (+Had) | 0.75 | 42.3% | 67% (65%) | 0.695 |
| VQ d16, 256 코드 (+Had) | 0.50 | 58.2% | 87% (86%) | 0.579 |

해석:
- 가중치 NMSE가 가우시안 rate-distortion 하한(2^-2R: 1.5bpw 12.5%, 1bpw 25%, 0.5bpw 50%)에 가깝다. 가중치를 독립 값으로 보는 압축(RTN, 단순 VQ)은 **1bit 근처에서 이론적으로 품질을 지킬 수 없다**.
- Hadamard 회전은 RTN의 출력 오차를 30% 안팎 줄이지만, VQ와 1bit 미만에서는 효과가 거의 없다.
- **RAM 용량과 품질이 정면으로 부딪힌다.** routed expert 544B를 RAM 약 75GB에 다 담으려면 평균 약 1.1bpw가 상한인데, 단순 방법으로 출력 오차를 수% 안에 두려면 2~3bpw가 필요하다(2.25bpw면 153GB).
- 1bit 아래로 가려면 가중치의 구조를 써야 한다: (1) expert 간 중복(공유 기저, C2), (2) 활성값 방향만 맞추는 출력 기준 압축(GPTQ류), (3) 증류로 회복(L2·L4). 아니면 (4) 일부 expert를 SSD 계층에 두거나 가지치기로 용량 자체를 줄인다.
### 모델 전체 품질 (2026-11-09, `scripts/quant_eval.py`, 평가 16K 토큰, routed expert만 압축)

| 설정 | ppl | KL (상위 64, nats) | top-1 일치 |
|---|---|---|---|
| 원본 | 3.970 | 0 | 100% |
| 같은 가중치, fp32 계산 경로 (대조군) | 3.970 | **0.033** | **95.4%** |
| RTN 3bit + Hadamard (레이어 10 MoE NMSE 2.8%) | 3.728 | 0.125 | 90.6% |
| RTN 2bit + Hadamard (12.5%) | 3.741 | 0.331 | 83.6% |

- **측정 잡음 바닥이 높다.** 가중치가 같아도 GEMM 누적 순서만 바뀌면 KL 0.033, top-1 4.6%가 바뀐다. 미세한 차이가 top-k 라우팅을 뒤집고 40층을 지나며 커지는 것으로 본다. 압축 손실은 이 바닥과 비교한다.
- **ppl은 이 모델에서 품질 지표로 쓸 수 없다.** 압축하면 ppl이 오히려 6% 낮아졌고, 대조군으로 계산 경로 버그가 아님을 확인했다. KL·top-1(이후 과제 정확도)로 판단한다.
- 3bit+Had는 바닥의 3.8배, 2bit+Had는 10배. **단순 RTN으로는 3bit도 공짜가 아니고 2bit는 손실이 크다.** RAM 상한(평균 약 1.1bpw)까지 가려면 출력 기준 압축, 공유 기저, 레이어별 민감도에 따른 혼합 정밀도, SSD 계층 중 무엇이든 필요하다.

## 5. 학습 (로컬 전용)

| ID | 내용 | 비용 [추정] |
|---|---|---|
| L0 | Teacher 데이터 생성: Pass 1(8.1절)에서 층 단위로 N 토큰을 통과시켜 top-k logit 저장 | N=50만 기준 hidden state 40GB, logit 1GB 미만 |
| L2 | 다층 슬라이딩 블록 증류 (연속 2~4층 공동 최적화) | 층당 원본 약 14GB, 로컬 가능 |
| L4 | **오프로딩 end-to-end 증류**: 학생(약 50GB)을 RAM에 두고 층 단위로 GPU에 흘려 codebook·scale·Q4 보정항만 KL로 학습 | 스텝당 수~수십 초, 하루 수천만 토큰 규모. Phase 1에서 실측 |
| L3 | DSpark drafter 도메인 자기 증류 (채택률 향상, 무손실) | drafter만 학습, 로컬 가능 |

L4가 클라우드 증류를 대신하는 품질의 핵심 지렛대다. 비용 추산은 실측 전 가설이다.

## 6. 실행

| ID | 내용 | 근거 |
|---|---|---|
| R1 | 캐시 밖 expert는 CPU에서 바로 계산, PCIe로는 hidden state만 이동 | [논문] Fiddler, KTransformers |
| R2 | AVX-512 LUT GEMV, codebook/LUT L3 상주 | [논문] T-MAC |
| R3 | 적응형 top-k | [논문] Ada-K |
| R4 | 캐시 조건부 라우팅 (손실, 토글) | [논문] Skliar et al. |
| R5 | 활성 희소성으로 down_proj 열 건너뛰기 | [논문] TEAL |
| R6 | 다음 층 router 선적용으로 expert 예측 | [논문] Pre-gated MoE |
| R7 | RAM 대역폭 기준 "CPU 계산 vs GPU 승격" 스케줄러 | [아이디어] |
| R8 | gate 값에 비례한 bit-plane 읽기량 | [아이디어] |
| R9 | 공유 기저 `B·x`는 GPU에서 층당 1회, CPU는 Δ만 | [아이디어] |
| R10 | n-gram 문맥 → expert 기록표로 미리 읽기 | [아이디어] |
| R11 | Engram은 원본 safetensors를 SSD에서 직접 조회, 자주 쓰는 행은 RAM 캐시 | [논문] Engram |
| R12 | 긴 prefill은 expert를 층 단위로 GPU에 스트리밍하며 큰 배치로 처리 (40층 전부 필요) | [논문] 일반 기법 |
| T1 | GPU attention 중 다음 층 예측 expert를 L3로 prefetch (RAM 유휴 제거) | [아이디어] |
| T2 | 상주형 GPU 메가커널 + pinned 메모리 메일박스 (WDDM 동기화 지연 제거) | [논문] 메가커널, Mirage MPK |
| T3 | 다음 블록 draft(GPU)와 현재 블록 검증(CPU) 병렬 | [논문] PEARL |
| T4 | 함께 활성화되는 expert를 메모리상 인접 배치 | [아이디어] |

## 7. Speculative decoding

| ID | 내용 | 근거 |
|---|---|---|
| S1 | 내장 DSpark/MTP 블록 draft + 검증 (무손실) | [확인] config |
| S2 | draft는 GPU 캐시 expert로만 라우팅 | [아이디어] |
| S3 | expert 합집합이 작은 가지 우선 검증 | [아이디어] |
| S4 | prompt lookup / n-gram draft | [논문] |
| S5 | confidence head 출력으로 검증 길이를 토큰마다 조절 (신뢰도 낮은 꼬리는 검증하지 않음 → expert 합집합 축소) | [확인] 모듈 존재 + [아이디어] 활용 |

다중 토큰 검증은 공식 코드에 없으므로 직접 구현한다. window ring buffer, Compressor의 미완성 그룹 상태, 압축 KV, indexer 캐시, Engram 해시 캐시를 거절된 토큰만큼 되돌리는 롤백이 필요하다.

## 8. Windows 구현 방침

- **레퍼런스 실행**: transformers 5.22.2에는 `deepseek_v41`이 없다(`deepseek_v4`까지만 있음). 그래서 공식 `inference/model.py`와 `kernel.py`를 순수 PyTorch로 옮긴 단일 GPU 레퍼런스를 네이티브 Windows(Python 3.14, torch 2.15.0+cu134, `.venv`)에서 돌린다. WSL 배포판은 모두 제거했다(기능만 남김).
- **엔진**: 네이티브 C++/CUDA + AVX-512. Triton은 Windows 지원이 불안정하므로 CUDA 커널을 직접 작성한다.
- **OS 설정**: HAGS 켜기, "메모리에 페이지 잠금" 권한 + `MEM_LARGE_PAGES`, 고성능 전원 계획, 스레드 고정(`SetThreadAffinityMask`), 모델 폴더 Defender 실시간 검사 제외.
- **I/O**: Engram은 IoRing 또는 unbuffered overlapped I/O.
- **동기화**: T2 메가커널로 WDDM 지연을 상쇄한다.
- **저장 위치**: 모델 관련 파일은 C:(NVMe)에 둔다. H:도 같은 NVMe 위의 VHDX라 공간을 공유한다. 원본 510GB를 통째로 받지 않는다(8.1절).
- **다운로드**: HF 파일은 CDN(`us.aws.cdn.hf.co`, 이 PC에서는 AWS 싱가포르로 조회)에서 받는다. 사용자의 지역 차단 규칙은 bf6.exe 전용(`BF6 region block`)으로 바꿨으므로 다운로드에 영향이 없다. 원본은 `C:\models\DeepSeek-V4.1-Flash`에 리비전 `314113f8`로 고정해 보관한다(사용자 승인, 실험 반복용).

## 8.1 디스크 최소화 파이프라인

원칙: **원본은 디스크에 쓰지 않는다.** HTTP range 요청으로 텐서를 RAM에 직접 받아 처리한다. 한 번의 순차 패스(Pass 1)에서 teacher 계산, 라우팅 통계, 압축을 모두 끝낸다. safetensors 헤더로 각 텐서의 byte offset을 알 수 있어서 필요한 범위만 받을 수 있고, 실패하면 그 범위만 다시 받는다.

1. embed → 층 0 → … → 층 39 → head → MTP 순서로 받는다. 층 하나(7.4GB)는 RAM에 들어가고, 다음 층은 처리 중에 미리 받는다.
2. 층 L마다 다음을 수행한다.
   - 보정 토큰 N개의 teacher hidden state를 층 L에 통과시킨다.
   - 라우팅 통계(빈도, 연속 재사용률, gate 분포, 다음 층 예측 정확도)를 기록한다.
   - 층 L을 압축한다(C1~C5, Q1, Q4, L2). 입력은 teacher 스트림과 student 스트림을 모두 쓴다.
   - student 층을 저장하고 원본 샤드를 지운다.
3. Engram 테이블은 해당 층(1, 14) 차례 전에 조각 단위로 받아 C8 PQ로 바로 인코딩한다. codebook은 먼저 받은 표본 행으로 학습한다. 보정 토큰이 실제로 조회하는 행은 FP8 원본 값도 보관해서, 같은 패스에서 PQ 대비 KL을 잰다.
4. 마지막 층 다음에 teacher top-k logit(N 토큰)을 저장한다. L4 증류는 이것만으로 진행한다.

층 사이의 의존성(KV 원천 층 2/8/14/20, indexer, candidate 층 20)도 hidden state와 함께 보관해야 한다. 크기는 작다.

| 항목 | 용량 [추정] |
|---|---|
| hidden state (mHC 4×5120 bf16) | 토큰당 40KB. N=50만이면 스트림당 20GB, teacher+student 40GB. RAM 우선, 넘치면 디스크 |
| 처리 중인 층 + 미리 받는 층 | RAM 약 15GB, 디스크 0 |
| Engram 원본 | 조각 단위로 RAM을 거쳐 감, 디스크 0 |
| **최종 = 피크** | student expert 약 50GB + dense/attention 약 4GB + drafter 약 2GB + Engram PQ 12~25GB + teacher logit 1GB 미만 → **약 70~85GB** (hidden state가 RAM을 넘칠 때만 추가) |

C8 PQ 품질이 나쁘면 Engram을 4bit(약 110GB)로 후퇴한다.

대가:
- 압축 설정을 처음부터 다시 하려면 원본을 다시 받아야 한다(디스크 대신 네트워크 비용). 층 단위로 재처리할 수 있어서 일부 층만 다시 받는 것도 가능하다.
- Pass 1 동안 RAM을 많이 쓴다(층 15GB + hidden state 40GB). 다른 앱은 정리해야 한다.
- L4에 쓰는 teacher 토큰 수가 Pass 1의 N으로 제한된다. 부족하면 Pass 2(재다운로드)로 늘린다.
- 층당 압축 계산 시간(특히 벡터 양자화 탐색)이 길 수 있다. 층 0에서 실측하고 N과 기법을 조정한다.

## 9. 속도 추산 [추정]

| | Plan A | Plan B | **Plan C** |
|---|---|---|---|
| 평균 bpw | 1.6 | 0.8 | 약 0.9 (사용 빈도 가중) |
| 예상 tok/s | 약 35~40 | 약 55~65 | **약 45~70** |

## 9.1 Phase 0 라우팅 실측 (2026-11-09, `data/routing_v1_report.json`)

보정 데이터 104K 토큰(범용 9개 분류). 캐시 내용은 짝수 시퀀스로 고르고 적중률은 홀수 시퀀스에서 decode 순서로 시뮬레이션했다.

| 항목 | 실측 | 설계에 주는 의미 |
|---|---|---|
| expert 사용 편중 | 레이어별 상위 10% expert가 활성의 33.5%, 상위 20%가 48.5% (균등이면 10%, 20%) | 학습 때 부하 균형이 걸려 있어 **정적 편중이 약하다**. 사용 빈도 기반 고정 캐시·비트 배분(Q1, C6)의 이득은 예상보다 작다 |
| 연속 토큰 expert 재사용 | 38% (무작위 1.6%) | **시간 지역성이 강하다**. 동적 캐시가 핵심 |
| GPU 캐시 적중률 (전체 15,360개 중 용량) | 고정 / LRU / 절반 고정+LRU: 930개 26% / **62%** / 54%, 1,600개 34% / **72%** / 65%, 2,500개 44% / **80%** / 74% | **LRU가 가장 좋다**. 고정 캐시를 섞으면 오히려 나빠진다 |
| 분류별 고정 캐시(1,600) 적중률 | 한국어 52%, 수학 43%, 대화 36%, 영어 웹 21%, 원시 코드 17% | 분류마다 쓰는 expert가 꽤 다르다 |
| 라우팅 가중치 (순위 1~6) | 30 / 21 / 16 / 13 / 11 / 10% | 비교적 평평하다. 90% 질량에 평균 5.46개, 80%에 4.54개 → **적응형 top-k(R3)의 이득은 작다** (손실 없이 줄일 여지 약 10%) |
| 미리 읽기 예측 (한 레이어 앞, 예측기 A) | 상위 6개 69%, 12개 83%, 24개 90% | **R6 유효**. 단 Engram 직후 레이어 15(30%)와 2(39%)는 약하다 |
| 미리 읽기 예측 (같은 레이어 attention 시점, B) | 68% / 81% / 89% | A와 비슷하고 미리 알 수 있는 시간이 짧으므로 A를 쓴다 |
| 학습 절반에서 한 번도 안 쓰인 expert | 16개 / 15,360개 | 사용 기록만으로 가지치기(C1)할 여지는 거의 없다. 품질 영향 측정이 필요하다 |

**LRU 캐시 기준 decode 추산 [추정]** (상시 경로 VRAM 약 3.8GB, 캐시 약 7GB, 캐시 미스는 PCIe DMA 56GB/s로 GPU에 올려 계산, 미리 읽기로 지연은 숨긴다고 가정):

| | expert 크기 | 캐시 용량 | LRU 적중 | 토큰당 미스 전송량 | 전송 + 상시 경로 | 이론 상한 → 현실치 |
|---|---|---|---|---|---|---|
| Plan A (1.6bpw) | 7.1MB | 약 990개 | 약 62% | 약 650MB | 11.5 + 4.2ms | 64 → **약 35 tok/s** |
| Plan C (약 0.9bpw) | 4.0MB | 약 1,750개 | 약 73% | 약 260MB | 4.6 + 4.2ms | 110 → **약 55~60 tok/s** |

speculative decoding(S1)은 이 위에 곱해진다. 연속 토큰 재사용이 38%라서 검증 토큰 수에 비해 expert 합집합은 덜 커진다.

전제(이전): RAM은 실측값(CPU 50GB/s, CPU+DMA 혼합 58GB/s), GPU 캐시 적중 35~45%, DSpark 단계당 약 3토큰 채택. 대역폭은 측정이 끝났고, 적중률과 채택률은 Pass 1 라우팅 통계 전까지 가설이다. 동기화 비용은 launch+sync 9.4µs × 토큰당 약 80회 ≈ 0.75ms로 예상보다 작았고, T2 메일박스(1.1µs)로 더 줄일 수 있다.

## 10. 평가 기준

- 품질: 고정 평가 세트에서 FP4 원본 대비 토큰별 KL, top-1 일치율, 소형 과제 점수.
  - 잠정 목표: top-1 일치율 90% 이상, 평균 KL 0.1 nats 이하. Phase 1 초기에 재조정.
- 속도: 4K 문맥 decode tok/s, prefill tok/s, 토큰당 RAM 읽기 바이트.

## 11. 위험

1. 1bit 미만 품질 붕괴 → Q1 배분 + L4 증류로 완화, 실패 시 Plan A로 후퇴.
2. 신규 아키텍처(mHC, CSA2, Engram, DSpark)를 직접 구현 → 모듈별로 레퍼런스와 출력 일치 테스트를 먼저 만든다.
3. 적중률·CPU 커널 효율·채택률 가정이 틀릴 수 있음 → Phase 0 측정으로 확정.
4. 디스크 → 8.1절 range 스트리밍으로 피크와 최종 모두 약 70~85GB. Engram PQ가 실패하면 약 110GB가 더해진다.
5. 공식 커널(tilelang)과 분산 코드는 Windows에서 그대로 못 쓴다 → `kernel.py`의 수치를 순수 PyTorch로 그대로 옮긴 단일 GPU 레퍼런스를 만든다. WSL은 제거했으므로 tilelang과 직접 비교하지 않고, (a) 커널별로 수학적 정의와 대조하는 단위 테스트, (b) 실제 가중치로 알려진 텍스트의 perplexity와 생성 결과 확인으로 검증한다.
6. 다중 토큰 검증과 롤백을 직접 구현해야 한다(7절).

## 12. 단계

0. 측정: 대역폭, 레퍼런스 실행, 라우팅 통계.
1. 압축 파이프라인 + 로컬 증류(L0, L2, L4).
2. 런타임 v1 (R1, R2).
3. 캐시·예측·스케줄러 (C7, R3, R6, R7, T1, T2).
4. Speculative (S1~S3, T3, L3).
5. 손실 기법 토글 실험 (R4, R5, R8, Q3, C4).
