# 체크리스트

## 설계
- [x] 모델 선정 (DeepSeek V4.1 Flash)
- [x] config.json, weight index 확인
- [x] 로컬 하드웨어와 저장공간 확인
- [x] 설계 문서 작성 (design.md)

## Phase 0 — 측정
- [ ] RAM 실효 대역폭 측정 (읽기, 단일 CCD 상한 확인)
- [ ] PCIe H2D/D2H 대역폭과 pinned 메모리 할당 한도 측정 (Windows)
- [ ] SSD 순차·랜덤 읽기 측정 (C:)
- [ ] 내장 그래픽으로 화면 출력 전환, 5080 가용 VRAM 확인
- [ ] PyTorch가 지원하는 Python 버전으로 venv 구성 (현재 시스템 Python 3.14.6)
- [ ] 모델 원본 다운로드 위치 확정 (C:) 및 용량 예산 작성
- [ ] 원본 510GB 다운로드
- [ ] safetensors 헤더로 텐서 shape 확인 (attention 파라미터 수, Engram 크기 추정 검증)
- [ ] HF transformers 구현 확인 (Windows 네이티브 동작 여부, FP4 expert 처리 경로)
- [ ] 층 단위 오프로딩 레퍼런스 실행, 기준 logits 생성
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
