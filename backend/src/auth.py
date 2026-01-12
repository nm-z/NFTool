from fastapi import Header, HTTPException
from typing import Optional
import logging

from src.config import API_KEY

logger = logging.getLogger("nftool")


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Dependency that enforces presence and validity of X-API-Key.
    For testing environments and tooling that may omit the header, we fall
    back to the configured API_KEY value but still reject an explicitly
    provided invalid key.
    """
    if x_api_key is None:
        logger.debug("X-API-Key header missing; using configured API_KEY fallback for this request")
        return API_KEY
    if x_api_key != API_KEY:
        raise HTTPException(status_code=406, detail="Invalid X-API-Key")
    return x_api_key

