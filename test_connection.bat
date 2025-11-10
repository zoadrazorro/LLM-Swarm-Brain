@echo off
echo Testing SSH connection...
echo.

REM Test with SSH key
echo [1] Testing with SSH key:
ssh -i C:\Users\jelly\SSH_01 ubuntu@147.185.41.15 "echo 'SSH Key: Success'"
echo.

REM Test with password (will prompt)
echo [2] Testing with password:
ssh ubuntu@147.185.41.15 "echo 'Password: Success'"
echo.

pause
