# 체크리스트

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
- [ ] T3 SSD 계층까지 포함한 배분 (RAM 예산 + SSD 대역폭 제약)
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
