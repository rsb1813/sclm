@echo off
rem Builds the Phase 0 benchmarks into bench\build.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /O2 /arch:AVX512 /std:c++20 /EHsc membw.cpp /Fobuild\ /Febuild\membw.exe || exit /b 1
cl /nologo /O2 /std:c++20 /EHsc ssdbench.cpp /Fobuild\ /Febuild\ssdbench.exe || exit /b 1
cl /nologo /O2 /arch:AVX512 /std:c++20 /EHsc lowbit_gemv.cpp /Fobuild\ /Febuild\lowbit_gemv.exe || exit /b 1
nvcc -O2 -std=c++20 -arch=sm_120 -Xcompiler "/EHsc /O2 /utf-8" pcie.cu -o build\pcie.exe || exit /b 1
nvcc -O2 -std=c++20 -arch=sm_120 -Xcompiler "/EHsc /O2 /utf-8 /arch:AVX512" concurrent.cu -o build\concurrent.exe || exit /b 1
