from abstract_utilities import *

abs_dir = "/run/user/1000/gvfs/sftp:host=192.168.1.100,user=solcatcher/srv/hugpy/src/hugpy/py"#get_initial_caller_dir()

funcs={}

dirs,files = get_files_and_dirs(abs_dir,allowed_exts=".toml",excluded_dirs=['tests','build'])

for file in files:
    input(file)
    content = read_from_file(file)
    lines = content.split('\n')
    functions =  [line.split('(')[0].split(' ')[1] for line in lines if line.startswith('def ')]
    for function in functions:
        if function not in funcs:
            funcs[function] = []
        funcs[function].append(file)

        
for func,paths in funcs.items():
    if len(paths) >1:
        print(func)
        for path in paths:
            print(path)
        input()
