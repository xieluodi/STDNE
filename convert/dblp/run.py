import os
import platform

sys = platform.system()

if sys == 'Windows':
    if os.path.exists('orca.exe'):
        pass
    else:
        os.system('g++ -O2 -std=c++11 -o orca.exe orca.cpp')
    for i in range(27):
        os.system('orca.exe 5 {}.txt {}.out'.format(i, i))
elif sys == 'Linux':
    if os.path.exists('orca'):
        pass
    else:
        os.system('g++ -O2 -std=c++11 -o orca orca.cpp')
    for i in range(27):
        os.system('orca 5 {}.txt {}.out'.format(i, i))
