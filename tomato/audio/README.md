In order to check the memory usage of a function use the decorator @profile at the beginning of the function.

To get the memory usage graph run: 

mprof run test_memory.py
mprof plot

(need to install memory-profiler)
pip install -U memory_profiler
