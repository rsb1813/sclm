# 컨텍스트 노트

## 2026-11-08

- **모델로 DeepSeek V4.1 Flash 선택.** Kimi K3.1 Pro(활성 104B)와 GLM-5.3(활성 40B)보다 토큰당 읽는 바이트가 가장 적다. 단점은 신규 아키텍처라 기존 런타임이 없다는 것.
- **config.json에서 확인한 사실**: expert가 원본부터 FP4, routed expert가 파라미터의 대부분(약 544B), MTP/DSpark drafter 내장, Engram 테이블 존재. HF 샤드 헤더는 샌드박스가 CDN(us.aws.cdn.hf.co) 접속을 막아 받지 못했다. attention 파라미터와 Engram 크기는 역산한 추정치다.
- **병목은 RAM 대역폭(약 55GB/s)으로 판단.** DMA도 같은 대역폭을 쓰므로 "CPU에서 계산 vs GPU 승격"을 하나의 비용 모델로 다룬다.
- **Plan C 채택**: 평균 bpw를 깎기보다 사용 빈도 가중 정밀도, RAM 유휴 제거, 증류로 품질을 확보한다.
- **사용자 제약: 클라우드 금지, Windows에서 완결.** 클라우드 end-to-end 증류(L1)는 제외하고 로컬 오프로딩 end-to-end 증류(L4)로 대체했다. 실현 가능성은 Phase 1에서 실측한다.
- **레퍼런스는 네이티브 Windows로.** WSL2에서 C:를 읽으면 9P 경유라 느리고, ext4 VHDX에 510GB를 복사할 공간(C: 여유 753GB)도 없다. WSL2는 공식 코드가 Linux 전용 커널을 요구할 때만 보조로 쓴다.
- **저장 위치**: 모델은 C:(NVMe)에 둔다. H:는 ReFS 150GB 가상 디스크라 코드와 소형 산출물만 둔다.
- **Triton 미사용**: Windows 지원이 불안정해서 CUDA 커널을 직접 작성한다.
