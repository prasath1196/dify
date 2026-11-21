class LLMError(ValueError):
    """Base class for all LLM exceptions."""

    description: str | None = None

    def __init__(self, description: str | None = None):
        self.description = description


class LLMBadRequestError(LLMError):
    """Raised when the LLM returns bad request."""

    description = "Bad Request"


class ProviderTokenNotInitError(ValueError):
    """
    Custom exception raised when the provider token is not initialized.
    """

    description = "Provider Token Not Init"

    def __init__(self, *args: object, description: str | None = None):
        super().__init__(*args)
        if description is not None:
            self.description = description
        elif args:
            self.description = str(args[0])


class QuotaExceededError(ValueError):
    """
    Custom exception raised when the quota for a provider has been exceeded.
    """

    description = "Quota Exceeded"


class AppInvokeQuotaExceededError(ValueError):
    """
    Custom exception raised when the quota for an app has been exceeded.
    """

    description = "App Invoke Quota Exceeded"


class ModelCurrentlyNotSupportError(ValueError):
    """
    Custom exception raised when the model not support
    """

    description = "Model Currently Not Support"
