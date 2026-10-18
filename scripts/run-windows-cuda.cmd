@echo off
rem Runs a sob command on Windows with NVIDIA's toolchain set up, in this order:
rem   1. MSVC (vcvars64.bat), which torch.compile needs for CUDA graphs;
rem   2. the CUDA toolkit, which TileLang needs, added AFTER vcvars so its PATH survives;
rem   3. TileLang told to use MSVC, not clang-cl (tile-ai/tilelang#3509).
rem Usage, from the repository root:
rem   scripts\run-windows-cuda.cmd reproduce --tag v0.1.0 --machine rtx4060-laptop --note "..."
rem   scripts\run-windows-cuda.cmd reproduce --tag v0.1.0 --machine rtx4060-laptop --resume results\rtx4060-laptop\<run>
setlocal

if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

if not exist "%VCVARS%" (
  echo vcvars64.bat not found at "%VCVARS%". Set VCVARS to your Visual Studio's copy.
  exit /b 1
)
call "%VCVARS%" >nul
where cl >nul 2>nul || (echo MSVC cl.exe is not on PATH after vcvars64.bat & exit /b 1)

set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TILELANG_DISABLE_CLANG_CL=1"
set "HF_HUB_OFFLINE=1"

uv run sob %*
