@echo OFF

: # Here's where I dispatch a bunch of PODs (whatever I have on hand).
echo Hello, and welcome, to this batch of POD testing pods.

:start
echo Running through pods now.
call python pod.py
call wait 30
call pypy pod.py
call wait 30
call python312 pod.py
call wait 30
call python311 pod.py
call wait 30
call python310 pod.py
call wait 30
call python39 pod.py
call wait 30
call python38 pod.py
call wait 30

echo Press <CTRL> + <C> to stop, or press <ENTER> to do again
pause
goto :start

echo Now this is pod racing
