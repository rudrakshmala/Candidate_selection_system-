import os
import subprocess
import time

def main():
    start_time = time.time()
    
    # We can't easily measure peak memory cross-platform without psutil, 
    # but we can just run the command and use standard python timing.
    cmd = ["python", "src/rank.py", "--candidates", "data/candidates.jsonl", "--features", "artifacts/features.parquet", "--out", "submission.csv"]
    
    # Start process
    proc = subprocess.Popen(cmd)
    proc.communicate()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
