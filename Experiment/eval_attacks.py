import os
import time

startTime = time.time()

for a in range(10):
    os.system(f"python run-sd-attacks.py")

executionTime = (time.time() - startTime)
print('Execution time in minutes: ' + str(executionTime / 60))
