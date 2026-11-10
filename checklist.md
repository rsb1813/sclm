# 체크리스트

## 지금
- 하는 일: 런타임 설계안 비교 (저장한 부호를 읽어 토큰 생성, Phase 2)
- 최근 결과: tier20v100 (RAM 1.00bpw) KL 0.0956, top-1 91.4%, 부호 63.3GiB 저장 완료
- 대기: 보정 247시퀀스 평가 (요청 시), 생성 시험판 (요청 시)

## 설계
- [x] 모델 선정 (DeepSeek V4.1 Flash)
- [x] config.json, weight index 확인
- [x] 로컬 하드웨어와 저장공간 확인
- [x] 설계 문서 작성 (design.md)

## Phase 0 — 측정
- [x] RAM 실효 대역폭 측정 (CPU 50~54GB/s, CCD 상한 확인)
- [x] CPU 읽기 + DMA 동시 대역폭 측정 (합계 58~59GB/s)
- [x] PCIe H2D/D2H 대역폭과 pinned 메모리 할당 한도 측정 (56GB/s, 48GiB 성공)
- [x] CPU-GPU 동기화 지연 측정 (launch+sync 9.4µs, mapped ping-pong 1.1µs)
- [x] SSD 순차·랜덤 읽기 측정 (7.4GB/s, QD32 53만 IOPS)
- [ ] 내장 그래픽으로 화면 출력 전환, 5080 가용 VRAM 재확인 (사용자 작업)
- [x] venv 구성 (`.venv`, Python 3.14.6, torch 2.15.0+cu134, transformers 5.22.2, 5080 동작 확인)
- [x] 공식 추론 코드 확보와 분석 (`ref/deepseek_v41`, git 제외). transformers에 `deepseek_v41` 없음
- [x] Docker 종료와 시작 프로그램 등록 제거, WSL 종료, 전원 계획 고성능 → RAM 가용 54.1 → 67.9GB
- [ ] 실행 직전 Chrome(약 11GB), Steam, EA, Epic 등 정리 (사용자 작업)
- [x] 디스크 용량 예산 작성 (design.md 8.1, 피크 약 200~230GB)
- [x] 샤드 구성 확인 (층당 7.4GB, Engram 203GB)
- [ ] 다운로드 방식 결정 (CDN이 샌드박스에서 차단됨 → 사용자 실행 또는 승인)
- [ ] range 요청으로 safetensors 헤더 읽기 (텐서 offset·shape 확인, 사용자 실행 또는 승인)
- [x] 순수 PyTorch 레퍼런스 작성 (`sclm/reference/kernel.py`, 공식 파일은 수정 없이 반입)
- [x] 커널별 단위 테스트 (수학적 정의와 대조) + 작은 무작위 모델 end-to-end 실행 테스트 (14개 통과)
- [x] Engram 해시 검증 (압축 어휘 99,092, 소수 구간 합 = 테이블 행 수)
- [x] 실제 가중치로 perplexity 확인 (레퍼런스 정확성의 최종 검증). 생성(decode) 경로는 층 단위 실행기로는 비현실적으로 느려서 런타임 단계에서 확인
- [x] 실제 가중치 로더 (`sclm/checkpoint.py`: wo_a FP8→bf16 역양자화, expert int8 → fp4 view, Engram 테이블은 memmap 행 조회)
- [x] 층 단위 실행기 (`sclm/layerwise.py`: Block 하나씩 생성·로드·해제) — 레이어 0 실제 가중치 엄격 로드·실행 통과
- [x] 방화벽 정리: dropship 제거, bf6.exe 전용 `BF6 region block` 규칙 생성 (다운로드와 무관해짐)
- [ ] 원본 전체 다운로드 (46/48 완료, Engram 샤드 2개 진행 중)
- [x] `python -m scripts.ppl`로 영어·한국어·코드 perplexity 확인: 2.87 / 2.54 / 1.37 (모두 128토큰 미만이라 window 경로만 검증됨)
- [x] 긴 문맥(2,048토큰) 정상 vs 압축 경로 차단 비교: 256토큰 이후 ppl 3.07 vs 12.07 → 압축 KV·indexer·candidate 경로 정상
- [x] 범용 보정 데이터 구성 (`data/calib_v1`, 104K 토큰, 9개 분류)
- [x] VRAM 상한 코드 검증·커밋, 수집기 블록 참조 누수 수정, Engram 병렬 읽기
- [x] 라우팅 기록 (18.4분, 104K 토큰 × 40층) → 분석 (design.md 9.1)
- [x] bf16 커널 변경 후 perplexity 재확인: 영어 2.874→2.859, 한국어 2.537→2.506, 코드 1.367→1.370 (±1%, 누적 순서 차이)

## Phase 1 — 압축 (진행 중)
- [x] 압축 함수 모듈 (`sclm/quant.py`: RTN, Hadamard, k-means++ VQ) + 테스트
- [x] 레이어 10 압축 방법 비교 (`scripts.quant_probe`): RTN 3bit+Had 2.8%, 2bit+Had 12.5%, VQ 1bpw 51% (design.md 4.1)
- [x] MoE 출력 오차 ↔ 모델 전체 KL 관계 측정: 잡음 바닥 KL 0.033, 3bit+Had 0.125, 2bit+Had 0.331 (design.md 4.1)
- [ ] 평가 세트를 분류별로 고르게 다시 구성 (지금 8개에는 다국어·도구 호출 없음)
- [ ] 레이어별 민감도 측정 (한 레이어씩 2bit로 낮춰 KL 변화) → 혼합 정밀도 근거
- [x] expert 간 중복 측정 → 활성 부분공간 사영은 모델 전체에서 실패 (KL 1.98~3.40, 거대 활성값이 NMSE를 가림)
- [x] GPTQ 레이어 10: 2bit NRD −22% (+0.36bit), 3bit −16%; 편향 제거 β는 H 기준이어야 함; NRD 가중 Hessian 이득 없음
- [x] GPTQ 모델 전체 평가: 2bit KL 0.295→0.242 (초과 −20%, +0.19bit 상당), ppl은 반대로 4.14→4.55 (design.md 4.2)
- [x] T1 축소 편향 측정: expert 출력 β 0.90(3bit) / 0.70(2bit), SwiGLU 곱셈 누적 확인 (design.md 4.2)
- [x] T1 판별: 편향 제거로 초과 KL −11~12% (2bit 0.328→0.295, 3bit 0.125→0.114), ppl 이상 현상의 원인 확인
- [x] T2 NRD 검증: 레이어 10에서 KL과 순위 상관 0.93 (에너지 NMSE는 역순), RTN과 사영을 갈라 냄
- [x] T3 두 자원 배분 계산: 사용 빈도 가중 +0.5bit 상당, 20 tok/s 기준 RAM expert 1.61~1.74bpw
- [x] T3 배분기 구현과 검증: 2.75bpw 배분 KL 0.126 ≈ 균일 3bit(3.25bpw) 0.114 → +0.4bit 상당
- [x] T3 SSD 계층까지 포함한 배분 (`scripts/allocate_tiers.py`): VRAM 캐시 370개로 20 tok/s에서 D 0.081 (RAM 0.95bpw) / 0.059 (1.10bpw), 캐시 없으면 0.225 / 0.173
- [x] T6 판별: 압축 모델은 덜 확신하지만 전역 온도로 되돌리면 KL이 커짐 (최적 배율 0.9, −1%) → 마지막 단 온도 보정 없음
- [x] 저비트 선택지 GPTQ 측정 (레이어 10, g128): NRD 1bit 0.765 (RTN 1.276), 3값 0.532, 2bit 0.421, 3bit 0.231
- [x] GPTQ 편향 제거 기준 비교: 없음이 모든 비트에서 최선 (1bit NRD 0.765→0.647), 층 공분산 기준은 최악 (출력 1.14~1.67배)
- [x] tier20 모델 전체 평가: RAM 0.95bpw + SSD 원본 + VRAM 캐시로 KL 0.143, top-1 89.2% (균일 약 3bpw 상당), en_web·code_raw가 약점
- [x] 층별 민감도 (8층씩 5그룹, RTN 2bit): 초과 KL이 그룹 간 가산, 가운데 층 1.5배 민감하나 층 간 재배분 이득 약 1.3%
- [x] 전역 배분 (RAM·SSD를 층 간 공유 + 민감도 가중): 민감도 가중 D 0.0790 → 0.0719 (−9%)
- [x] T8 GPTQ 뒤 step 최소제곱 재적합: 모든 비트에서 NRD 변화 0.3% 미만 → 기각
- [x] 보정 데이터 양: 토큰 두 배마다 GPTQ NRD −7~10%, expert당 75토큰에서 포화 안 됨
- [x] 데이터 부족 보완 Hessian (expert별 축소 β, 라우터 이웃 토큰): 이득 0.3% 이하 또는 악화 → 기각, 고정 β=0.5 유지
- [x] T7 경계 헤지: KL −0.003~0.004(dense 0.0329→0.0304, RTN3 0.1138→0.1099)에 expert 읽기 +13~15% → 기각
- [x] SSD 계층 캐시 정책 재생: LRFU(+전체 이력, bypass)가 LRU보다 읽기 5~10% 적음, MIN 대비 격차는 8토큰 넘는 미래 정보라 예측으로 못 메움 (design.md T3)
- [x] tier20g (전역 배분) + 보정 43시퀀스 모델 전체 평가: KL 0.143 → 0.121, top-1 90.2% (초과 −20%, 균일 약 3.17bpw 상당, 예상 0.117)
- [x] CPU 계층 GEMV 커널 벤치 (`bench/lowbit_gemv.cpp`, A·B 공통): 8스레드 49.5GB/s로 RAM 대역폭 상한, tier20g 비율로 토큰당 27.7ms (design.md 9.2)
- [x] T9 GPTQ 안의 벡터 반올림 (`quant.gptq_vq`): 행 스케일 벡터 부호가 스칼라 형식과 같은 D를 0.12~0.22bpw 적게 씀 (1.0bpw 0.387, 2.0bpw 0.136, 3.0bpw 0.045)
- [x] 벡터 부호 CPU 커널 벤치 (`bench/vq_gemv.cpp`): 네 형식 모두 8스레드 50~52GB/s로 RAM 대역폭 상한, int8 표 오차 1.5e-4~4.7e-4
- [x] 벡터 부호 배분 예측 (tier20v, tier20v110): 민감도 가중 왜곡 0.0719 → 0.0516 (0.95bpw) / 0.0386 (1.10bpw)
- [x] 기각: 고유기저 가변 비트, 뉴런 중요도 비트, w2 비대칭 보정, w2 우선 비트, SSD 계층 손실 저장 (design.md T9)
- [x] v2c6 (6비트 부호) 커널과 w2 표 생성 비용 실측: 51.0GB/s, w2 표 7~18µs (expert당 계산 44~260µs)
- [x] 보정 파이프라인 재작성: 보정 입력 단일 CPU 버퍼, 층 단위 이어하기(`--checkpoint-every`), 층마다 RAM 가용량 출력, 벡터 형식 실행(`quant.VQLinear`)
- [x] 앞 2층 스모크 테스트: tier20v 오차 0.0035 (tier20g 0.0043), 이어하기 차이는 실행 간 잡음(3.7e-4)과 같은 크기
- [x] 부호를 디스크에 저장 (`--save-codes`, `sclm/codes.py`): 층당 safetensors 하나, CPU 커널의 타일 배치, 파일만으로 평가 스트림 재현(오차 0)
- [x] tier20v 모델 전체 평가 (보정 43시퀀스, 65.9분): KL 0.121 → 0.100, top-1 90.2% → 91.0% (초과 −24%, 예측 −28%)
- [x] 247시퀀스 실행이 RAM 부족으로 죽는 원인 수정: Engram mmap 페이지가 작업 집합에 쌓임(시퀀스당 0.23GiB) → 조회마다 view를 열고 닫음
- [x] 평가 단축 (GPTQ 역행렬 일괄 계산, w1·w3 한 번에, 보정 배치의 expert 계산을 expert 단위로 미룸): tier20v 보정 43시퀀스 65.9분 → 40.2분, KL 0.1001 → 0.0989, top-1 91.0% → 91.1% (실행 간 편차 안), VRAM 최대 10.8 → 12.8GiB
- [ ] 보정 247시퀀스 tier20v 모델 전체 평가 (층 1에서 RAM 부족으로 중단됨, 수정 후 재실행 대기)
- [x] tier20v110 (RAM 1.10bpw) 모델 전체 평가 (보정 43시퀀스): KL 0.0804, top-1 91.8%, ppl 3.998 (예측 KL 약 0.08). 잠정 목표를 실행 간 편차 밖에서 통과, RAM 계층 약 75GB
- [x] tier20v100 (RAM 1.00bpw) 모델 전체 평가와 부호 저장 (보정 43시퀀스, 41.3분): KL 0.0956, top-1 91.4%, ppl 4.026 (예측 KL 약 0.092보다 나쁨), 부호 63.3GiB (`C:\models\sclm-codes\tier20v100`)
- [x] 중간 예산 배분 예측: 1.00bpw 왜곡 0.0468 (KL 약 0.092, RAM 63.3GiB), 1.05bpw 0.0425 (KL 약 0.086, 66.4GiB)
- [x] GPTQ VRAM 최대 12.8 → 9.9GiB: 스케일 재조정 전에 작업 텐서 해제, 양자화가 끝난 expert는 층이 끝날 때까지 CPU에 둠 (tier20v110 층 19에서 CUDA OOM 두 번)
- [x] SSD 계층 + VRAM LRU 캐시: 드문 expert는 몰아서 쓰여 적중 81%, Che 근사는 낙관적이라 정확한 LRU 재생으로 예산 검증
- [x] 라우팅 고정 실험: 잡음 바닥은 64%가 뒤집힘, 2bit 손실은 88%가 expert 출력 오차 (design.md 4.2)
- [ ] RAM 용량(평균 약 1.1bpw 상한) vs 품질(2~3bpw 필요) 충돌 해소안 결정: 출력 기준 압축 / 공유 기저 / SSD 계층 / 가지치기
- [ ] 층 단위 레퍼런스 실행 코드 작성 (Pass 1 골격, 층 간 KV/indexer 의존성 포함)
- [ ] 층 0에서 압축 계산 시간 실측, N과 기법 조정
- [ ] 평가 세트 고정 (도메인 프롬프트)
- [ ] 라우팅 로그 수집: expert 빈도 편중, 연속 토큰 재사용률, 다음 층 예측 정확도, gate 분포
- [ ] 측정값으로 속도 추산(design.md 9절) 갱신

## Phase 1 — 압축과 증류
- [ ] C1 가지치기
- [ ] C3 양자화 파이프라인
- [ ] Q1 사용 빈도 가중 비트 배분
- [ ] L0 teacher 데이터 생성
- [ ] L2 다층 블록 증류
- [ ] L4 오프로딩 end-to-end 증류 실측
- [ ] 품질 평가와 Plan A/B/C 판정

## Phase 2~5
- [ ] Phase 2 런타임 v1 (R1, R2)
- [ ] Phase 3 캐시·예측·스케줄러
- [ ] Phase 4 speculative
- [ ] Phase 5 손실 기법 토글 실험

## 도구
- [x] 상태 조회 화면 (`status.bat`, `scripts/status.py`): 실행 중인 작업, GPU·RAM·디스크, 품질 결과, 단계별 진행, 최근 커밋. 맨 위 "지금" 절을 그대로 보여 주므로 하는 일이 바뀌면 그 절을 고친다
