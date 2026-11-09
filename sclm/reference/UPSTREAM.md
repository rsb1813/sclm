# 출처

`model.py`, `engram.py`, `vision.py`, `image_processor.py`, `config.json`, `LICENSE`는 아래 저장소의 `inference/`에서 **수정 없이** 가져온 파일이다(MIT).

- 저장소: `deepseek-ai/DeepSeek-V4.1-Flash` (Hugging Face)
- 커밋: `314113f8a7dadb388215396ba5fbee2c610a04d6` (2026-10-16)

`kernel.py`만 직접 작성했다. 원본은 tilelang 커널이라 Windows에서 돌지 않으므로, 같은 수치를 순수 PyTorch로 옮겼다. 함수 이름과 인자는 원본과 같아서 `model.py`가 그대로 import한다. 이 폴더를 `sys.path`에 넣고 쓴다(`tests/conftest.py` 참고).

원본이 갱신되면 위 파일들을 그대로 덮어쓰고 `kernel.py`의 인터페이스가 바뀌었는지만 확인한다.
