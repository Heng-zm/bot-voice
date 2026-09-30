@echo off
chcp 65001 >nul
echo ==================================================================
echo  Bot Voice Zip Uploader - Anajak Cloud (my.anajak.cloud:2022)
echo ==================================================================
echo Connecting to chuokimheng.2852ceb0@my.anajak.cloud:2022 ...
echo Enter your Anajak Cloud account password when prompted:
echo.
sftp -P 2022 -o StrictHostKeyChecking=no -b "%~dp0upload-zip.sftp" chuokimheng.2852ceb0@my.anajak.cloud
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================================
    echo  UPLOAD ZIP COMPLETED SUCCESSFULLY!
    echo ==================================================================
    echo Next steps:
    echo 1. Go to https://my.anajak.cloud
    echo 2. In Files, click '...' next to bot-voice-update.zip and select 'Unarchive'
    echo 3. In Console, click 'Restart'
) else (
    echo.
    echo SFTP upload failed. Please verify your password and try again.
    echo Tip: Your SFTP password is the same password you use to log in to https://my.anajak.cloud
)
