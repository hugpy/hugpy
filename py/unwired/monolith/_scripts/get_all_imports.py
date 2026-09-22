from abstract_utilities import *
lines = trace_all_imports("/srv/hugpy/src/abstract_hugpy_dev/src/abstract_hugpy_dev/__init__.py").split('\n')
parents = {}
for line in lines:
    if line.startswith("from abstract_hugpy_dev.") and line.endswith("*"):
        child = line.split('.')[1]
        if child not in parents:
            parents[child] = []
        parents[child].append(line)
input(parents)

