import sys
from typing import Callable, Any


def run_process(function: Callable[[], Any], cleanup: Callable[[], Any] = None):
    try:
        function()
    except (KeyboardInterrupt, SystemExit):
        if cleanup is not None:
            print("Process Interrupted, Cleaning Up!")
            cleanup()
        else:
            print("Process Interrupted, Exiting!")
            sys.exit(0)
