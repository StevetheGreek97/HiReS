# HiReS/ui/run_pipeline_subprocess.py
import json
import sys
from HiReS.source.pipeline import Pipeline
from HiReS.source.config import Settings

"""
This script:
1. Receives a JSON-encoded Settings dict as stdin.
2. Runs the pipeline normally.
3. Prints "./PROGRESS {fraction} {message}" lines for UI.
4. Prints "DONE" at the end.
"""

def print_progress(frac, msg):
    sys.stdout.write(f"PROGRESS {frac:.4f} {msg}\n")
    sys.stdout.flush()

def main():
    # Read JSON settings from stdin
    raw = sys.stdin.read()
    cfg_dict = json.loads(raw)
    cfg = Settings(**cfg_dict)

    pipeline = Pipeline(cfg)

    def callback(frac, msg):
        print_progress(frac, msg)

    result = pipeline.run(
        workers=1,
        debug=False,
        progress_cb=callback,
    )

    sys.stdout.write("DONE\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
