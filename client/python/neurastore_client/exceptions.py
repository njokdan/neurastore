"""Exception hierarchy for the NeuraStore client.

A client library's job is to translate transport-level failures (HTTP
status codes, connection errors) into meaningful, catchable domain
exceptions -- not to let `requests.exceptions.HTTPError` leak through
directly, which would force every caller to know NeuraStore's HTTP
status code conventions instead of just catching a NeuraStore-specific
exception.
"""


class NeuraStoreError(Exception):
    """Base class for all errors raised by this client. Catch this to
    handle any NeuraStore-related failure without worrying about which
    specific subtype it was."""


class ConnectionError(NeuraStoreError):
    """Could not reach the server at all -- it's not running, the host/
    port is wrong, or the network is down. Distinct from a server that
    responded with an error status, which means the server IS reachable."""


class NotFoundError(NeuraStoreError):
    """The requested record does not exist (HTTP 404)."""


class BadRequestError(NeuraStoreError):
    """The request was malformed or invalid per the server's own
    validation (HTTP 400) -- e.g. an empty vector, or querying before
    the index has been built."""


class ServerError(NeuraStoreError):
    """The server encountered an internal error while processing an
    otherwise valid request (HTTP 500). Usually indicates a bug or a
    resource problem on the server side, not a client mistake."""
