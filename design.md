# SCLM 설계 — DeepSeek V4.1 Flash 로컬 고속 추론

표기: **[확인]** config/weight index/로컬 측정으로 확인 · **[추정]** 그 수치로 계산 · **[논문]** 발표된 기법 · **[아이디어]** 미발표 가설

## 1. 목표와 제약

- 목표: 단일 사용자 decode **20~60 tok/s**, 품질은 FP4 원본 대비 최소 손실.
- 제약: **클라우드 사용 금지**, **Windows 11에서 완결**(네이티브 + WSL2 허용).
- 하드웨어 [확인]

| 자원 | 사양 | 실효 대역폭 |
|---|---|---|
| GPU | RTX 5080 16GB, PCIe 5.0 x16, NVFP4 | VRAM 약 900GB/s, PCIe 약 50GB/s [추정] |
| RAM | 96GB (16+32 혼합 4장) DDR5-4200 듀얼채널 | 약 55~60GB/s [추정, Phase 0 측정] |
| CPU | Ryzen 7 9800X3D 8C/16T, AVX-512, L3 96MB | 단일 CCD 읽기 상한 약 64GB/s [추정] |
| SSD | Solidigm P44 Pro 2TB (C:, 여유 753GB) | 약 7GB/s 순차 |
| 기타 | H: 150GB ReFS 가상 디스크(프로젝트), D: HDD, WSL2 Ubuntu 24.04, CUDA 13.4 | |

**전체 병목은 RAM 대역폭이다.** RAM→GPU DMA도 같은 대역폭을 소비하므로 expert의 GPU 승격은 재사용될 때만 이득이다.

## 2. 모델 사실

- [확인] hidden 5120, 40층, routed expert 384(top-6) + shared 1, expert 중간 차원 2304.
- [확인] 원본 dtype: expert FP4, 나머지 FP8 (block 32×32, ue8m0 scale). 체크포인트 총 510GB.
- [확인] scoring `sqrtsoftplus`, `noaux_tc`, routed_scaling 1.5, mHC(`hc_mult` 4), sliding window 128, 압축·희소 attention(`compress_ratios`, indexer).
- [확인] KV 원천 층 `[2, 8, 14, 20]`, Engram 층 `[1, 14]`, MTP 3층(DSpark: block 5, 128 experts top-3, markov head).
- [추정] expert 1개 35.4M → routed 총 543.6B. decode 토큰당 routed 8.5B + 상시(attention·shared·head) 약 7B ≈ 15.6B.
- [추정] prefill은 앞쪽 약 20층만 필요(YOCO식 KV 재사용).
- [추정] Engram 테이블 약 200GB, 토큰 id로 주소 결정, 토큰당 약 12KB 조회.

## 3. 메모리 계층

```
GPU 16GB : 상시 경로(NVFP4/3bit) ~3.8GB · DSpark ~2.5GB · hot expert 캐시 ~6.5GB · KV/작업 ~1.5GB
RAM 96GB : 압축 routed expert 전체 (Plan C 약 50GB) · codebook/LUT는 L3 상주
SSD (C:) : Engram 원본 그대로 · FP4 원본(teacher, 압축 원천)
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
| L0 | Teacher 데이터 생성: FP4 원본을 SSD에서 큰 배치로 스트리밍해 top-k logit 저장 | 패스당 약 45~60s, 패스당 16~32K 토큰 → 하루 수천만 토큰 |
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
- **저장 위치**: 모델 원본은 C:(NVMe)에 둔다. H:는 150GB라 코드와 소형 산출물만 둔다. 원본 510GB를 받으면 C: 여유가 약 243GB로 줄어드므로 압축 산출물과 teacher 데이터 용량을 관리해야 한다.

## 9. 속도 추산 [추정]

| | Plan A | Plan B | **Plan C** |
|---|---|---|---|
| 평균 bpw | 1.6 | 0.8 | 약 0.9 (사용 빈도 가중) |
| 예상 tok/s | 약 40 | 약 60~70 | **약 50~75** |

전제: RAM 55GB/s, GPU 캐시 적중 35~45%, DSpark 단계당 약 3토큰 채택. 모두 Phase 0 측정 전 가설이다.

## 10. 평가 기준

- 품질: 고정 평가 세트에서 FP4 원본 대비 토큰별 KL, top-1 일치율, 소형 과제 점수.
  - 잠정 목표: top-1 일치율 90% 이상, 평균 KL 0.1 nats 이하. Phase 1 초기에 재조정.
- 속도: 4K 문맥 decode tok/s, prefill tok/s, 토큰당 RAM 읽기 바이트.

## 11. 위험

1. 1bit 미만 품질 붕괴 → Q1 배분 + L4 증류로 완화, 실패 시 Plan A로 후퇴.
2. 신규 아키텍처(mHC, CSA2, Engram, DSpark)를 직접 구현 → 모듈별로 레퍼런스와 출력 일치 테스트를 먼저 만든다.
3. 적중률·CPU 커널 효율·채택률 가정이 틀릴 수 있음 → Phase 0 측정으로 확정.
4. 디스크 여유 부족 → 산출물 용량 예산을 Phase 0에서 확정.

## 12. 단계

0. 측정: 대역폭, 레퍼런스 실행, 라우팅 통계.
1. 압축 파이프라인 + 로컬 증류(L0, L2, L4).
2. 런타임 v1 (R1, R2).
3. 캐시·예측·스케줄러 (C7, R3, R6, R7, T1, T2).
4. Speculative (S1~S3, T3, L3).
5. 손실 기법 토글 실험 (R4, R5, R8, Q3, C4).
