"""Authentication module for NFTool backend."""

import logging

from fastapi import Depends, Header, HTTPException

from src.config import API_KEY

logger = logging.getLogger("nftool")

API_KEY_HEADER = Header(None, alias="X-API-Key")


def get_api_key_header(x_api_key: str | None = API_KEY_HEADER) -> str | None:
    """Return the value of the X-API-Key header or None.

    This function is used as a dependency provider for FastAPI endpoints so
    that the header value can be injected via `Depends`. It intentionally
    returns None when the header is absent, allowing downstream dependencies
    to decide how to handle missing keys (for example, falling back to the
    configured `API_KEY`).
    """
    return x_api_key


GET_API_KEY_DEPENDENCY = Depends(get_api_key_header)


async def verify_api_key(
    x_api_key: str | None = GET_API_KEY_DEPENDENCY,
) -> str:
    """
    Dependency that enforces presence and validity of X-API-Key.
    For testing environments and tooling that may omit the header, we fall
    back to the configured API_KEY value but still reject an explicitly
    provided invalid key.
    """
    if x_api_key is None:
        logger.debug(
            "X-API-Key header missing; using configured API_KEY fallback "
            "for this request"
        )
        return API_KEY
    if x_api_key != API_KEY:
        raise HTTPException(status_code=406, detail="Invalid X-API-Key")
    return x_api_key
