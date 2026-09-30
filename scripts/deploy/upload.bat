@echo off
chcp 65001 >nul
echo ==================================================================
echo  Bot Voice SFTP Uploader - Anajak Cloud (my.anajak.cloud:2022)
echo ==================================================================
echo Connecting to chuokimheng.2852ceb0@my.anajak.cloud:2022 ...
echo Enter your Anajak Cloud account password when prompted:
echo.
sftp -P 2022 -o StrictHostKeyChecking=no -b "%~dp0upload.sftp" chuokimheng.2852ceb0@my.anajak.cloud
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================================
    echo  UPLOAD TO ANAJAK CLOUD COMPLETED SUCCESSFULLY!
    echo ==================================================================
    echo Next step: Go to https://my.anajak.cloud and click 'Restart' on your server console.
) else (
    echo.
    echo SFTP upload failed. Please verify your password and try again.
)

