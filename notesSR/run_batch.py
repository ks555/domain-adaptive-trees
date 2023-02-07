# -*- coding: utf-8 -*-
"""
Launch batch.py in parallel (on multi-core machines)
"""
import subprocess
import utils

# template of calling batch.py
template = 'python batch.py {} {} subset1'

processes = []
step = 1 # smaller -> more processes
for i in range(0, len(utils.states), step):
    command = template.format(i, i+step)
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
# wait all child processes
output = [p.wait() for p in processes]