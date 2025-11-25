from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
import bcrypt
import os
import logging
import secrets

logger = logging.getLogger(__name__)

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security: Minimum key length requirements per algorithm (in bytes)
# HS256 requires minimum 256 bits (32 bytes) per RFC 7518 Section 3.2
# We enforce 32 bytes minimum, recommend 64+ bytes for better security
MIN_KEY_LENGTH_HS256 = 32  # 256 bits minimum
RECOMMENDED_KEY_LENGTH_HS256 = 64  # 512 bits recommended


def validate_jwt_secret_key(secret_key: str, algorithm: str, environment: str) -> bool:
    """
    Validate JWT secret key strength to prevent weak encryption vulnerabilities.
    
    This addresses CVE-2025-45768 (PyJWT weak encryption) by enforcing minimum
    key length requirements per RFC 7518 and NIST SP800-117 standards.
    
    Args:
        secret_key: The JWT secret key to validate
        algorithm: The JWT algorithm (e.g., 'HS256')
        environment: Environment name ('production' or 'development')
    
    Returns:
        True if key is valid, False otherwise
    
    Raises:
        ValueError: If key is too weak (in production) or invalid
    """
    if not secret_key:
        return False
    
    # Calculate key length in bytes (UTF-8 encoding)
    key_bytes = len(secret_key.encode('utf-8'))
    
    # Validate based on algorithm
    if algorithm == "HS256":
        min_length = MIN_KEY_LENGTH_HS256
        recommended_length = RECOMMENDED_KEY_LENGTH_HS256
        
        if key_bytes < min_length:
            error_msg = (
                f"JWT_SECRET_KEY is too weak for {algorithm}. "
                f"Minimum length: {min_length} bytes (256 bits), "
                f"Current length: {key_bytes} bytes. "
                f"This violates RFC 7518 Section 3.2 and NIST SP800-117 standards."
            )
            if environment == "production":
                logger.error(f"❌ SECURITY ERROR: {error_msg}")
                raise ValueError(error_msg)
            else:
                logger.warning(f"⚠️  WARNING: {error_msg}")
                logger.warning("⚠️  This key is too weak and should not be used in production!")
                return False
        
        if key_bytes < recommended_length:
            logger.warning(
                f"⚠️  JWT_SECRET_KEY length ({key_bytes} bytes) is below recommended "
                f"length ({recommended_length} bytes) for {algorithm}. "
                f"Consider using a longer key for better security."
            )
        else:
            logger.info(f"✅ JWT_SECRET_KEY validated: {key_bytes} bytes (meets security requirements)")
    
    return True


# Load and validate JWT secret key
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

# Security: Require JWT_SECRET_KEY in production, allow default in development with warning
if not SECRET_KEY or SECRET_KEY == "your-secret-key-change-in-production":
    if ENVIRONMENT == "production":
        import sys
        logger.error("JWT_SECRET_KEY environment variable must be set with a secure value in production")
        logger.error("Please set JWT_SECRET_KEY in your .env file or environment")
        logger.error("Generate a secure key: python -c \"import secrets; print(secrets.token_urlsafe(64))\"")
        sys.exit(1)
    else:
        # Development: Use a default but warn
        SECRET_KEY = "dev-secret-key-change-in-production-not-for-production-use"
        logger.warning("⚠️  WARNING: Using default JWT_SECRET_KEY for development. This is NOT secure for production!")
        logger.warning("⚠️  Please set JWT_SECRET_KEY in your .env file for production use")

# Validate key strength (addresses CVE-2025-45768)
try:
    validate_jwt_secret_key(SECRET_KEY, ALGORITHM, ENVIRONMENT)
except ValueError as e:
    # In production, validation failure is fatal
    if ENVIRONMENT == "production":
        import sys
        logger.error(f"❌ JWT_SECRET_KEY validation failed: {e}")
        logger.error("Generate a secure key: python -c \"import secrets; print(secrets.token_urlsafe(64))\"")
        sys.exit(1)


class JWTHandler:
    """Handle JWT token creation, validation, and password operations."""

    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = REFRESH_TOKEN_EXPIRE_DAYS

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(
        self, token: str, token_type: str = "access"
    ) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != token_type:
                logger.warning(
                    f"Invalid token type: expected {token_type}, got {payload.get('type')}"
                )
                return None

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                logger.warning("Token has expired")
                return None

            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT error: {e}")
            return None

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        # Use bcrypt directly to avoid passlib compatibility issues
        # Bcrypt has a 72-byte limit, so truncate if necessary
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            # Use bcrypt directly to avoid passlib compatibility issues
            # Bcrypt has a 72-byte limit, so truncate if necessary
            password_bytes = plain_password.encode('utf-8')
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
            
            hash_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except (ValueError, TypeError, Exception) as e:
            logger.warning(f"Password verification error: {e}")
            return False

    def create_token_pair(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Create both access and refresh tokens for a user."""
        access_token = self.create_access_token(user_data)
        refresh_token = self.create_refresh_token(user_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }


# Global instance
jwt_handler = JWTHandler()
