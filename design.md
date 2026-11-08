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
- [추정] expert 1개 35.4M → routed 총 543.6B. decode 토큰당 routed 8.5B + 상시(attention·shared·head) 약 7B ≈ 15.6B.
- [추정] prefill은 앞쪽 약 20층만 필요(YOCO식 KV 재사용).
- [확인] 샤드 구성: 층마다 7.4GB 샤드 1개(dense+routed, 총 296GB), Engram 2개(층 1, 14, 각 101.5GB, 총 203GB), MTP 7.9GB, vision 2.3GB, embed/head 각 1.3GB.
- [추정] Engram 행 하나가 약 264B(FP8 256차원 + scale). 토큰당 수십 회 4KiB 랜덤 읽기 → QD32에서 0.1ms 수준.

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
| R12 | prefill은 앞쪽 20층만, expert를 GPU로 스트리밍하며 큰 배치 처리 | [추정] 구조 |
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

## 8. Windows 구현 방침

- **레퍼런스 실행**: 네이티브 Windows PyTorch + HF transformers 구현(`model_type: deepseek_v41`)을 층 단위 오프로딩으로 돌린다. WSL2는 C: 접근이 9P 경유라 느리고, ext4 VHDX에 510GB를 복사할 공간도 없다. 공식 코드가 Linux 전용 커널을 요구할 때만 WSL2를 보조로 쓴다.
- **엔진**: 네이티브 C++/CUDA + AVX-512. Triton은 Windows 지원이 불안정하므로 CUDA 커널을 직접 작성한다.
- **OS 설정**: HAGS 켜기, "메모리에 페이지 잠금" 권한 + `MEM_LARGE_PAGES`, 고성능 전원 계획, 스레드 고정(`SetThreadAffinityMask`), 모델 폴더 Defender 실시간 검사 제외.
- **I/O**: Engram은 IoRing 또는 unbuffered overlapped I/O.
- **동기화**: T2 메가커널로 WDDM 지연을 상쇄한다.
- **저장 위치**: 모델 관련 파일은 C:(NVMe)에 둔다. H:도 같은 NVMe 위의 VHDX라 공간을 공유한다. 원본 510GB를 통째로 받지 않는다(8.1절).
- **다운로드**: HF 파일은 CDN(`us.aws.cdn.hf.co`)에서 내려받는데, Claude 샌드박스에서는 이 호스트 접속이 막혀 있다. 다운로드 단계는 사용자가 직접 실행하거나 샌드박스 해제 승인이 필요하다.

## 8.1 디스크 최소화 파이프라인

원칙: **원본은 샤드 단위로 받아 처리하고 바로 지운다.** 한 번의 순차 패스(Pass 1)에서 teacher 계산, 라우팅 통계, 압축을 모두 끝낸다.

1. embed → 층 0 → … → 층 39 → head → MTP 순서로 샤드를 받는다. 다음 샤드는 처리 중에 미리 받는다.
2. 층 L마다 다음을 수행한다.
   - 보정 토큰 N개의 teacher hidden state를 층 L에 통과시킨다.
   - 라우팅 통계(빈도, 연속 재사용률, gate 분포, 다음 층 예측 정확도)를 기록한다.
   - 층 L을 압축한다(C1~C5, Q1, Q4, L2). 입력은 teacher 스트림과 student 스트림을 모두 쓴다.
   - student 층을 저장하고 원본 샤드를 지운다.
3. Engram 샤드는 해당 층(1, 14) 차례 전에 받는다. 4bit 또는 2bit로 양자화하고(필요하면 사용 빈도 기반 행 정리), 원본을 지운다. 품질은 같은 패스에서 FP8 대비 KL로 확인한다.
4. 마지막 층 다음에 teacher top-k logit(N 토큰)을 저장한다. L4 증류는 이것만으로 진행한다.

층 사이의 의존성(KV 원천 층 2/8/14/20, indexer, candidate 층 20)도 hidden state와 함께 보관해야 한다. 크기는 작다.

| 항목 | 용량 [추정] |
|---|---|
| hidden state (mHC 4×5120 bf16) | 토큰당 40KB. N=50만이면 스트림당 20GB, teacher+student 40GB. RAM 우선, 넘치면 디스크 |
| 처리 중인 층 샤드 + 미리 받는 샤드 | 약 15GB |
| Engram 원본 1개 (처리 중) | 101.5GB |
| **피크** | Engram 14층 처리 시점 약 200~230GB |
| **최종** | student expert 약 50GB + dense/attention 약 4GB + MTP 약 3GB + Engram 55~105GB → **약 110~160GB** |

대가:
- 압축 설정을 처음부터 다시 하려면 원본을 다시 받아야 한다(디스크 대신 네트워크 비용).
- L4에 쓰는 teacher 토큰 수가 Pass 1의 N으로 제한된다. 부족하면 Pass 2(재다운로드)로 늘린다.
- 층당 압축 계산 시간(특히 벡터 양자화 탐색)이 길 수 있다. 층 0에서 실측하고 N과 기법을 조정한다.

## 9. 속도 추산 [추정]

| | Plan A | Plan B | **Plan C** |
|---|---|---|---|
| 평균 bpw | 1.6 | 0.8 | 약 0.9 (사용 빈도 가중) |
| 예상 tok/s | 약 35~40 | 약 55~65 | **약 45~70** |

전제: RAM은 실측값(CPU 50GB/s, CPU+DMA 혼합 58GB/s), GPU 캐시 적중 35~45%, DSpark 단계당 약 3토큰 채택. 대역폭은 측정이 끝났고, 적중률과 채택률은 Pass 1 라우팅 통계 전까지 가설이다. 동기화 비용은 launch+sync 9.4µs × 토큰당 약 80회 ≈ 0.75ms로 예상보다 작았고, T2 메일박스(1.1µs)로 더 줄일 수 있다.

## 10. 평가 기준

- 품질: 고정 평가 세트에서 FP4 원본 대비 토큰별 KL, top-1 일치율, 소형 과제 점수.
  - 잠정 목표: top-1 일치율 90% 이상, 평균 KL 0.1 nats 이하. Phase 1 초기에 재조정.
- 속도: 4K 문맥 decode tok/s, prefill tok/s, 토큰당 RAM 읽기 바이트.

## 11. 위험

1. 1bit 미만 품질 붕괴 → Q1 배분 + L4 증류로 완화, 실패 시 Plan A로 후퇴.
2. 신규 아키텍처(mHC, CSA2, Engram, DSpark)를 직접 구현 → 모듈별로 레퍼런스와 출력 일치 테스트를 먼저 만든다.
3. 적중률·CPU 커널 효율·채택률 가정이 틀릴 수 있음 → Phase 0 측정으로 확정.
4. 디스크 → 8.1절 스트리밍 파이프라인으로 피크 약 200~230GB, 최종 약 110~160GB. Engram 양자화 품질이 나쁘면 최종 용량이 최대 203GB 늘어난다.

## 12. 단계

0. 측정: 대역폭, 레퍼런스 실행, 라우팅 통계.
1. 압축 파이프라인 + 로컬 증류(L0, L2, L4).
2. 런타임 v1 (R1, R2).
3. 캐시·예측·스케줄러 (C7, R3, R6, R7, T1, T2).
4. Speculative (S1~S3, T3, L3).
5. 손실 기법 토글 실험 (R4, R5, R8, Q3, C4).
