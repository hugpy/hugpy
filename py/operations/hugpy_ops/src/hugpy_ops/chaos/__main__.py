"""`python -m hugpy_ops.chaos` -> the runner CLI (same as `hugpy-chaos`)."""
import sys

from hugpy_ops.chaos.runner import main

if __name__ == "__main__":
    sys.exit(main())
