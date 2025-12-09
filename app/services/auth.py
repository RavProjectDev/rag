import json
import logging
from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
from rag.app.core.config import get_settings, AuthMode

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)  # Don't auto-raise error, we'll handle it based on mode


async def verify_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    Verify JWT token using Supabase JWKS and extract user_id.
    In dev mode, skips authentication and returns a dummy user_id.
    In prd mode, requires valid JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials (optional in dev mode)
        
    Returns:
        user_id: The user ID extracted from the JWT token (or "dev-user" in dev mode)
        
    Raises:
        HTTPException: 401 if token is invalid or missing in prd mode
    """
    settings = get_settings()
    
    # Dev mode: skip authentication
    if settings.auth_mode == AuthMode.DEV:
        user_id = "dev-user"
        logger.info(f"User logged in (DEV mode): user_id={user_id}")
        return user_id
    
    # PRD mode: require authentication
    if not credentials:
        logger.warning("No credentials provided in PRD mode")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_token", "message": "Authentication required"}
        )
    
    if not settings.supabase_url:
        logger.error("Supabase URL not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "auth_config_error", "message": "Authentication not configured"}
        )
    
    token = credentials.credentials
    
    try:
        # Construct JWKS URL from Supabase URL
        jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        
        # Create JWKS client
        jwks_client = PyJWKClient(jwks_url)
        
        # Get the signing key from the token
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and verify the token
        decoded_token = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],  # ECC P-256 uses ES256 (ECDSA with SHA-256)
            audience="authenticated",  # Supabase default audience
            options={"verify_exp": True, "verify_aud": True}
        )
        
        # Extract user_id from the token
        user_id = decoded_token.get("sub")
        
        if not user_id:
            logger.warning("Token missing 'sub' claim")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "invalid_token", "message": "Token missing user_id"}
            )
        
        logger.info(f"User logged in (PRD mode): user_id={user_id}")
        return user_id
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "token_expired", "message": "Token has expired"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "Invalid token"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "auth_error", "message": "Authentication failed"}
        )


def verify(event) -> tuple[bool, str]:
    try:
        question = event.get("question")
        if not question:
            return False, "body needs to include question"

        return True, question

    except json.JSONDecodeError:
        return False, "body needs to be in json format"
    except ValueError as ve:
        return False, str(ve)
    except Exception as e:
        return False, str(e)
