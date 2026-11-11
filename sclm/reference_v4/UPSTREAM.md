# 출처

`model.py`와 `config.json`은 아래 저장소의 `inference/`에서 **수정 없이** 가져온 파일이다(MIT).

- 저장소: `deepseek-ai/DeepSeek-V4-Flash-0731` (Hugging Face)
- 커밋: `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (2026-08-01)

커널은 `sclm/reference/kernel.py`(순수 PyTorch)를 같이 쓴다. 두 모델의 `kernel.py`는 함수 이름과 인자가 같다. `sclm/upstream.py`가 `SCLM_MODEL=v4`일 때 이 폴더를 `sys.path`의 앞에 넣어 `model`은 여기서, `kernel`은 `sclm/reference`에서 찾게 한다.

`fast_hadamard_transform.py`는 직접 작성했다. 원본이 import하는 같은 이름의 패키지(CUDA 확장)를 대신한다.
