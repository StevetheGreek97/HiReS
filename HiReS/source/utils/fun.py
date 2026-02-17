import random
import threading
from contextlib import contextmanager
from itertools import cycle
import sys
import time

PRED_JOKES = [
    "Asking the neurons nicely…",
    "Consulting the oracle",
    "Feeding pixels to the GPU",
    "Counting polygons by hand (just kidding)",
    "Aligning tensors spiritually",
    "Convincing the model this is a Daphnia",
    "Applying deep thoughts to shallow water",
    "Zooming into the planktonverse",
    "Blaming noise on biology",
    "YOLO-ing responsibly",
]

@contextmanager
def spinner(base_msg: str = "Predicting"):
    frames = cycle(["/", "-", "\\", "|"])
    stop = threading.Event()

    joke = random.choice(PRED_JOKES)


    def run():
        nonlocal joke
        while not stop.is_set():

            sys.stdout.write(
                f"\r{base_msg} {next(frames)}  {joke}"
            )
            sys.stdout.flush()
            time.sleep(0.92)

        # clear line
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()