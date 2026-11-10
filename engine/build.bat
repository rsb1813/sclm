@echo off
rem Builds the engine core (engine\build\sclm_core.dll): the RAM-tier CPU kernels, the SSD-tier reader and its cache policy.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /O2 /arch:AVX512 /std:c++20 /EHsc /LD core.cpp ssd.cpp /Fobuild\ /Febuild\sclm_core.dll || exit /b 1
