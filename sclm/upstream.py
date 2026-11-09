# Imports the vendored upstream inference modules (sclm/reference) under their own top-level names.
import os
import sys

_REF = os.path.join(os.path.dirname(__file__), "reference")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

import engram  # noqa: E402
import model  # noqa: E402

CONFIG_PATH = os.path.join(_REF, "config.json")
