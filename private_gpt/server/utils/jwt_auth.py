import jwt
from fastapi import HTTPException
from jwt import PyJWKClient

from private_gpt.server.utils.user import User
from private_gpt.settings.settings import settings


def _parse_token(authorization: str) -> str:
    """Parse the authorization header into our JWT.

    @param authorization:
    @return:
    """
    parts = authorization.split(" ")
    # Checking if the header is correctly formatted
    if len(parts) == 2 and parts[0] == "Bearer":
        token = parts[1]
        return token
    else:
        raise HTTPException(400, "Incorrectly formed jwt authorization header")


class JWTAuth:
    jwks_client: PyJWKClient

    def __init__(self):
        self.jwks_client = PyJWKClient(
            settings().server.jwt_auth.jwksUrl, cache_jwk_set=True
        )

    def validate_jwt(self, authorization: str) -> User | None:
        try:
            token = _parse_token(authorization)
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            data = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience="https://expenses-api",
                options={"require": ["exp", "iss", "sub"], "verify_signature": True},
            )
            return User(
                sub=data["sub"],
                allowed_ingest=bool(
                    data.get(settings().server.jwt_auth.ingest_claim, False)
                ),
            )
        except Exception:
            return None
