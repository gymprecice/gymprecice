import subprocess
 
def launch_subprocess(cmd, run_script="", env_dir=""):
    assert cmd in ['run'], \
        "Error: invalid shell command - supported commands: 'run'"

    subproc = subprocess.Popen([f"./{run_script}"], shell=True, cwd=f"{env_dir}")
    
    return subproc
  
if __name__ == '__main__':
    
    import_precice = False # if True, mpirun cannot be called within python script

    if import_precice:
        import precice  # this breaks mpirun


    proc = launch_subprocess("run", run_script="run.sh", env_dir=".")
