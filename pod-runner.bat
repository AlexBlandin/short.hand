@echo OFF

: # Here's where I dispatch a bunch of PODs (whatever I have on hand).
echo Hello, and welcome, to this batch of POD testing pods.
call pypy pod.py
call wait 30
call python pod.py
call wait 30
call python311 pod.py
call wait 30
