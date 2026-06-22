import schemathesis
from src.api import app, verify_api_key

# Bypass API Key for testing
app.dependency_overrides[verify_api_key] = lambda: "bypassed-for-testing"

schema = schemathesis.openapi.from_asgi("/openapi.json", app)

@schema.parametrize()
def test_api(case):
    case.call_and_validate()
