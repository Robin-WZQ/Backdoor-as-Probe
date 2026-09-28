"""Compatibility shim.

The target-response detector is the output-side target-response detector.  This module
keeps the old command name usable, but it delegates directly to the current
implementation and contains no input-trigger detector.
"""

from .target_response import main


if __name__ == "__main__":
    main()
