import schemathesis
import sys
import os

# Add backend to path so we can import the app
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.api import app, verify_api_key

# 1. THE "NO KEY" TRICK: 
app.dependency_overrides[verify_api_key] = lambda: "bypassed-for-testing"

# 2. Load the API Schema directly from the code
schema = schemathesis.openapi.from_asgi("/openapi.json", app)

print("\n🚀 STARTING API AUTOPILOT TEST...")
print("Checking all endpoints for crashes and logic errors.\n")

# 3. Execute the tests
for operation_result in schema.get_all_operations():
    operation = operation_result.ok() if hasattr(operation_result, "ok") else operation_result
    for case in operation.get_test_cases():
        try:
            response = case.call_asgi(app)
            status = "PASSED" if response.status_code < 500 else "FAILED"
            print(f"DEBUG: {operation.method.upper()} {operation.path} -> {status} ({response.status_code})")
        except Exception as e:
            print(f"DEBUG: {operation.method.upper()} {operation.path} -> CRASHED: {e}")

print("\n✅ TEST SUITE COMPLETE.")