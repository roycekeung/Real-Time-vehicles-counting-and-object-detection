date /t
@echo off
echo 'start running the Automatic vehicle counting and object detection'
"C:\Python37\python.exe" "main.py"
timeout 3
echo 'start running the integration'
"C:\Python37\python.exe" "integration.py"
timeout 3
echo 'start running excel solver'
cscript Run_macro.vbs
echo 'Done, finished all computation!'
PAUSE
