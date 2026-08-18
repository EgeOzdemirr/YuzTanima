@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ========================================
echo  Python yolu guncelleniyor...
echo ========================================
echo.

REM Python 3.13 ara
set PYTHON_EXE=
for %%P in (
    "C:\Program Files\Python313\python.exe"
    "C:\Program Files (x86)\Python313\python.exe"
) do (
    if exist %%P (
        set PYTHON_EXE=%%~P
        goto :bulundu
    )
)

REM Kullanici AppData altinda ara
for /f "tokens=*" %%A in ('dir /b /s "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" 2^>nul') do (
    set PYTHON_EXE=%%A
    goto :bulundu
)

REM where komutu ile ara
for /f "tokens=*" %%A in ('where python 2^>nul') do (
    "%%A" --version 2>&1 | findstr /C:"3.13" >nul
    if !errorlevel!==0 (
        set PYTHON_EXE=%%A
        goto :bulundu
    )
)

echo HATA: Python 3.13 bulunamadi!
echo Lutfen https://www.python.org/downloads/ adresinden Python 3.13 kurun.
echo.
pause
exit /b 1

:bulundu
echo Python 3.13 bulundu: %PYTHON_EXE%

REM Python dizinini al (exe olmadan)
for %%F in ("%PYTHON_EXE%") do set PYTHON_HOME=%%~dpF
REM Sondaki ters slash'i kaldir
if "%PYTHON_HOME:~-1%"=="\" set PYTHON_HOME=%PYTHON_HOME:~0,-1%

echo Python dizini: %PYTHON_HOME%
echo.

REM pyvenv.cfg guncelle
set CFG_FILE=3ddfav3\pyvenv.cfg
if not exist "%CFG_FILE%" (
    echo HATA: %CFG_FILE% bulunamadi!
    pause
    exit /b 1
)

REM Gecici dosyaya yaz
set TMP_CFG=%TEMP%\pyvenv_tmp.cfg
if exist "%TMP_CFG%" del "%TMP_CFG%"

for /f "tokens=*" %%L in ('type "%CFG_FILE%"') do (
    set LINE=%%L
    echo !LINE! | findstr /C:"home = " >nul
    if !errorlevel!==0 (
        echo home = %PYTHON_HOME%>> "%TMP_CFG%"
    ) else (
        echo !LINE! | findstr /C:"executable = " >nul
        if !errorlevel!==0 (
            echo executable = %PYTHON_EXE%>> "%TMP_CFG%"
        ) else (
            echo %%L>> "%TMP_CFG%"
        )
    )
)

copy /y "%TMP_CFG%" "%CFG_FILE%" >nul
del "%TMP_CFG%"

echo pyvenv.cfg guncellendi.
echo.
echo Simdi Baslat.exe veya Baslat.bat ile programi calistirabilirsiniz.
echo.
pause
