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
- **대역폭 실측 (`bench/`)**: CPU 읽기는 1스레드부터 50GB/s에서 막힌다(9800X3D 단일 CCD 연결 상한). 반면 pinned DMA는 56GB/s, 둘을 동시에 쓰면 합계 58~59GB/s가 나온다. 따라서 CPU 계산만 하는 것보다 CPU 계산과 GPU 승격을 섞는 R7이 유리하다는 실측 근거가 생겼다.
- **동기화 지연이 예상보다 작다**: launch+sync 9.4µs라 토큰당 80회여도 약 0.75ms. mapped 메모리 ping-pong이 1.1µs라 T2 메일박스 방식이 Windows에서도 유효하다.
- **측정 당시 RAM 가용량은 67GB**였다(백그라운드 앱이 약 29GB 점유). 실행 시 정리가 필요하다.
- **사용자 요청: 디스크 최소화.** 원본 510GB를 통째로 받지 않고, 층 샤드(7.4GB)를 하나씩 받아 Pass 1에서 teacher 계산·통계·압축을 끝낸 뒤 지운다. 피크는 Engram 원본 하나(101.5GB)가 있을 때 약 200~230GB, 최종은 약 110~160GB. 대가는 재압축 시 재다운로드와 teacher 토큰 수 제한(N)이다.
- **H:는 `C:\devdrive.vhdx`**라서 C:와 같은 NVMe이고 공간도 공유한다.
- **HF CDN이 샌드박스에서 차단된다.** API(huggingface.co)는 되지만 실제 파일(us.aws.cdn.hf.co)은 안 된다. 다운로드는 사용자가 직접 실행하거나 승인해야 한다.
