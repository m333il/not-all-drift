class InterpError(Exception):
    """Base exception for expected pipeline failures."""


class ConfigurationError(InterpError):
    """Raised when a resolved experiment configuration is unsafe or invalid."""


class ArtifactError(InterpError):
    """Raised for incompatible, corrupt, or incomplete run artifacts."""


class ParseError(InterpError):
    """Raised when a model response cannot be parsed under the output contract."""


class ReflectorUnavailable(BaseException):
    """Reflection endpoint failed too many times in a row.

    Not an ``InterpError``: GEPA swallows ordinary exceptions from the reflector.
    """
