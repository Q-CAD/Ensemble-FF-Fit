import os
import re
import subprocess
import sys
from glob import glob
from time import sleep

def submit_and_get_id(workdir, submission_file):
    """
    Submit with sbatch and parse the JobID from stdout.
    Returns an integer JobID, or None if submission failed.
    """
    orig_cwd = os.getcwd()
    os.chdir(workdir)
    print(f"Submitting job {os.path.join(workdir, submission_file)}")
    try:
        result = subprocess.run(
            ["sbatch", submission_file],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # Log the failure and stderr for diagnosis
        print(f"[ERROR] sbatch failed with exit {e.returncode}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr, file=sys.stderr)
        os.chdir(orig_cwd)
        sys.exit(1)
    finally:
        os.chdir(orig_cwd)

    # Typical stdout: "Submitted batch job 12345\n"
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        print(f"[ERROR] Could not parse JobID from sbatch output: {result.stdout!r}")
        return None

    return int(m.group(1))

def check_job(jobid, workdir):
    """
    Check the status of a single job:
      - in_queue: True if `squeue -j jobid` still lists it
      - done_flag: True if workdir/job.<jobid>.done exists
      - fail_flag: True if workdir/job.<jobid>.fail exists
    Returns a tuple (in_queue, done_flag, fail_flag).
    """
    # 1) Poll squeue for that job
    sq = subprocess.run(
        ["squeue", "-j", str(jobid), "-h"],
        capture_output=True,
        text=True
    )
    in_queue = bool(sq.stdout.strip())

    # 2) Look for the sentinel files
    done_flag = os.path.isfile(os.path.join(workdir, f"job.{jobid}.done"))
    fail_flag = os.path.isfile(os.path.join(workdir, f"job.{jobid}.fail"))

    # 3) Check for log files indicating a finished workflow
    all_complete = False
    logs = sorted(glob(os.path.join(workdir, 'logs*.txt')))
    if logs:
        with open(logs[-1]) as logfile:
            if 'EXITING WORKFLOW ENVIRONMENT' in logfile.read():
                all_complete = True

    return in_queue, done_flag, fail_flag, all_complete

def handling_logic(in_queue, done_flag, fail_flag, all_complete, jobid,
                   workdir, submission_file, resubmit, max_retries, 
                   retry_count, poll_interval):
    """
    Resubmission or exit logic.
    """
    if all_complete:
        print(f"Workflow for Job {jobid} finished; check subdirectories for job errors.")
        return False    
        
    elif in_queue:
        print(f"Job {jobid} still in queue...")
        return True

    # At this point, jobid is no longer in the queue
    elif done_flag:
        print(f"Job {jobid} exited the queue without failure.")
        if resubmit:
            print(f"Job {jobid} left queue without completing; resubmitting (attempt {retry_count+1})")
            return MatEnsemble_submission_wrapper(
                    workdir,
                    submission_file,
                    resubmit=True,
                    max_retries=max_retries,
                    retry_count=retry_count+1,
                    poll_interval=poll_interval
            )
        else:
            print(f"Exiting")
            return False
            
    elif fail_flag:
        print(f"Job {jobid} exited the queue with failure; check outputs. Exiting.")
        sys.exit(1)
            
    else:
        print(f"Job {jobid} not in queue.")
        return False

def clean_directory(directory):
    """
    Remove any old sentinel or log files before starting.
    """
    for pattern in ["job_record.txt", "job.*.done", "job.*.fail", 
                    "*.out", "*.err", "logs*.txt", "restart*.dat"]:
        for fn in glob(os.path.join(directory, pattern)):
            os.remove(fn)

def MatEnsemble_submission_wrapper(
    workdir,
    submission_file,
    resubmit=False,
    max_retries=3,
    retry_count=0,
    poll_interval=60):
    """
    Submit a SLURM job, monitor its lifecycle, and optionally resubmit.
    """
    if retry_count > max_retries:
        print("Maximum retries exceeded. Exiting.")
        sys.exit(1)

    # Step 0: check existing files
    unknown_id = 'unknown_id'
    in_queue, done_flag, fail_flag, all_complete = check_job(unknown_id, workdir)
    handling_logic(in_queue, done_flag, fail_flag, all_complete, unknown_id, 
                   workdir, submission_file, resubmit, max_retries, 
                   retry_count, poll_interval)
    
    # Step 1: cleanup old markers
    clean_directory(workdir)

    # Step 2: submit and grab JobID
    jobid = submit_and_get_id(workdir, submission_file)
    print(f"Submitted job {jobid}")

    # Step 3: poll until job leaves the queue
    run = True
    while run:
        sleep(poll_interval)
        in_queue, done_flag, fail_flag, all_complete = check_job(jobid, workdir)
        run = handling_logic(in_queue, done_flag, fail_flag, all_complete, jobid, 
                             workdir, submission_file, resubmit, max_retries, 
                             retry_count, poll_intervale)
