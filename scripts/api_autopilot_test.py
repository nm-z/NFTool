import importlib.util
import os
from typing import Any, cast

import schemathesis

# Dynamically load the backend FastAPI app from its source file so we don't need
# to modify sys.path at runtime and to keep imports at the top of the module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_API_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "backend", "src", "api.py"))
_spec = importlib.util.spec_from_file_location("nftool_api", _API_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load API module from {_API_PATH}")
_api_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api_module)
app = _api_module.app
verify_api_key = _api_module.verify_api_key

# 1. THE "NO KEY" TRICK
app.dependency_overrides[verify_api_key] = lambda: "bypassed-for-testing"

# 2. Load the API Schema directly from the code
schema = schemathesis.openapi.from_asgi("/openapi.json", app)

print("\nSTARTING API AUTOPILOT TEST...")
print("Checking all endpoints for crashes and logic errors.\n")

# 3. Execute the tests
for operation_result in schema.get_all_operations():
    operation_any = cast(Any, operation_result)
    operation = operation_any.ok() if hasattr(operation_any, "ok") else operation_any
    get_cases = getattr(operation, "get_test_cases", None)
    if not callable(get_cases):
        continue
    cases_iter = cast(Any, get_cases)()
    for case in cases_iter:
        try:
            response = case.call_asgi(app)
            status_code = getattr(response, "status_code", 0)
            status_str = "PASSED" if status_code < 500 else "FAILED"
            method = getattr(operation, "method", "UNKNOWN")
            path = getattr(operation, "path", "UNKNOWN")
            print(
                f"DEBUG: {method.upper()} {path} -> {status_str} "
                f"({getattr(response, 'status_code', 'n/a')})"
            )
        except Exception as exc:
            method = getattr(operation, "method", "UNKNOWN")
            path = getattr(operation, "path", "UNKNOWN")
            print(f"DEBUG: {method.upper()} {path} -> CRASHED: {exc}")

print("\nTEST SUITE COMPLETE.")
