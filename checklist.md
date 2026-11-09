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
- [ ] 실제 가중치로 perplexity·생성 결과 확인 (레퍼런스 정확성의 최종 검증)
- [ ] 실제 가중치 로더 (HF 이름 → 모델 이름, wo_a FP8→bf16 역양자화, expert int8 → fp4 view, MTP embed/head 공유)
- [ ] 층 단위 스트리밍 실행 (VRAM 16GB 안에서 층 하나씩 올려 전체 forward)
- [ ] 범용 보정 데이터 구성 (코딩, 한국어·영어 대화, 수학·추론, 다국어, 도구 호출)
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
