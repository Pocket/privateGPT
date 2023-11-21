"""Authentication mechanism for the API.

Define a simple mechanism to authenticate requests.
More complex authentication mechanisms can be defined here, and be placed in the
`authenticated` method (being a 'bean' injected in fastapi routers).

Authorization can also be made after the authentication, and depends on
the authentication. Authorization should not be implemented in this file.

Authorization can be done by following fastapi's guides:
* https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/
* https://fastapi.tiangolo.com/tutorial/security/
* https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/
"""
# mypy: ignore-errors
# Disabled mypy error: All conditional function variants must have identical signatures
# We are changing the implementation of the authenticated method, based on
# the config. If the auth is not enabled, we are not defining the complex method
# with its dependencies.
import logging
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from private_gpt.di import global_injector
from private_gpt.server.utils.user import User
from private_gpt.settings.settings import settings

# 401 signify that the request requires authentication.
# 403 signify that the authenticated user is not authorized to perform the operation.
NOT_AUTHENTICATED = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": 'Basic realm="All the API", charset="UTF-8"'},
)

logger = logging.getLogger(__name__)


def _simple_authentication(authorization: Annotated[str, Header()] = "") -> User:
    """Check if the request is authenticated."""
    if not secrets.compare_digest(authorization, settings().server.basic_auth.secret):
        # If the "Authorization" header is not the expected one, raise an exception.
        raise NOT_AUTHENTICATED
    return User(sub="basic_user", allowed_ingest=True)


def _jwt_authentication(authorization: Annotated[str, Header()] = "") -> User:
    from private_gpt.server.utils.jwt_auth import JWTAuth

    return global_injector.get(JWTAuth).validate_jwt(authorization)


if settings().server.jwt_auth.enabled:
    logger.debug("Using JWT based authentication for the request")

    # Method to be used as a dependency to check if the request
    # is authenticated for jwt auth.
    def authenticated(
        _jwt_authentication: Annotated[User | None, Depends(_jwt_authentication)]
    ) -> User:
        """Check if the request is authenticated."""
        assert settings().server.jwt_auth.enabled
        if _jwt_authentication is None:
            raise HTTPException(status_code=401, detail="Invalid JWT")
        return _jwt_authentication

elif settings().server.basic_auth.enabled:
    logger.debug("Using basic authentication for the request")

    # Method to be used as a dependency to check if the request
    # is authenticated for basic auth.
    def authenticated(
        _simple_authentication: Annotated[User | None, Depends(_simple_authentication)]
    ) -> User:
        """Check if the request is authenticated."""
        assert settings().server.basic_auth.enabled
        if _simple_authentication is None:
            raise NOT_AUTHENTICATED
        return _simple_authentication

else:
    logger.debug(
        "Defining a dummy authentication mechanism for fastapi, always authenticating requests"
    )

    # Define a dummy authentication method that always returns True.
    def authenticated() -> User:
        """Check if the request is authenticated."""
        return User(sub="unauthenticated_dummy_user", allowed_ingest=True)
