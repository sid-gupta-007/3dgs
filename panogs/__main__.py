"""Direct execution module for python -m panogs."""
import sys
from panogs.apps.cli import main

if __name__ == "__main__":
    sys.exit(main())
