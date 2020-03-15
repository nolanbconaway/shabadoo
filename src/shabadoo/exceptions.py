"""Custom errors for shabadoo models."""


class ShabadooException(Exception):
    """Parent class for exceptions."""

    def __str__(self):
        """Return the exception string."""
        name = self.__class__.__name__
        message = self.message or ""
        return f"{name}({message})"


class NotFittedError(ShabadooException):
    """Raised when using post-fit model functionality on an unfitted model."""

    def __init__(self, func=None):
        """Set the message."""
        name = func.__name__ if func else "function"
        self.message = f"Unable to call {name} before fitting model."


class AlreadyFittedError(ShabadooException):
    """Raised when calling fit on a fitted model."""

    def __init__(self, model):
        """Set the message."""
        name = model.__class__.__name__
        self.message = f"Model {name} has already been fitted!"


class IncompleteModel(ShabadooException):
    """Raised when initializing a model with missing config."""

    def __init__(self, model, attribute):
        """Set the message."""
        name = model.__class__.__name__
        self.message = f"Model `{name}` does not have attribute `{attribute}`!"


class IncompleteFeature(ShabadooException):
    """Raised when initializing a model with an incomplete feature."""

    def __init__(self, name, key):
        """Set the message."""
        self.message = f"Feature `{name}` does not have a {key}!"


class IncompleteSamples(ShabadooException):
    """Raised a model has incomplete samples for some reason."""

    def __init__(self, name):
        """Set the message."""
        self.message = f"No or not enough samples found for `{name}`."
