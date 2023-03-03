import subprocess
import os
 
def launch_subprocess(cmd, run_script="", env_dir=""):
    assert cmd in ['run'], \
        "Error: invalid shell command - supported commands: 'run'"
    
    subproc_env = {key: variable for key, variable in os.environ.items() if "MPI" not in key}  # fix line

    subproc = subprocess.Popen([f"./{run_script}"], shell=True, env=subproc_env, cwd=f"{env_dir}")
    
    return subproc
  
if __name__ == '__main__':
    
    import_precice = True # if True, mpirun cannot be called within python script

    if import_precice:
        import precice  # this breaks mpirun


    proc = launch_subprocess("run", run_script="run.sh", env_dir=".")
