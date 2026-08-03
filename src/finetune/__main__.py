"""Allow ``python -m finetune`` as well as the ``finetune`` console script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
