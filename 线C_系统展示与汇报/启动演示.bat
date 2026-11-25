@echo off
cd /d "%~dp0"
title 银行数据管理与异常交易检测 - 演示
set "UV=%USERPROFILE%\.cherrystudio\bin\uv.exe"
set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

echo ==========================================
echo   银行数据管理与异常交易检测 - 演示系统
echo   首次启动需安装依赖，请等待 1-2 分钟
echo   启动后浏览器会自动打开：
echo     http://localhost:8501
echo ==========================================
echo.

if exist "%UV%" goto use_uv
python -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_python
py -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_py
echo [错误] 未找到 Python 环境，请安装 Python 3.10+ 后运行：pip install -r requirements.txt
goto end

:use_uv
"%UV%" run --no-project --with streamlit --with pandas --with numpy --with lightgbm streamlit run streamlit_app.py
goto end

:use_python
python -m streamlit run streamlit_app.py
goto end

:use_py
py -m streamlit run streamlit_app.py

:end
echo.
echo 服务已停止，按任意键关闭窗口。
pause >nul
