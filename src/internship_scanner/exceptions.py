"""Application-specific exceptions."""


class InternshipScannerError(Exception):
    """Base class for expected scanner failures."""


class ConfigurationError(InternshipScannerError):
    """Raised when runtime configuration is invalid."""


class JobSourceError(InternshipScannerError):
    """Raised when jobs cannot be retrieved or decoded from a source."""


class IntelligenceError(InternshipScannerError):
    """Base class for expected AI intelligence failures."""


class IntelligenceCacheError(IntelligenceError):
    """Raised when the persistent inference cache cannot be accessed."""


class EmbeddingError(IntelligenceError):
    """Raised when an embedding provider cannot produce valid vectors."""


class LLMError(IntelligenceError):
    """Raised when an LLM provider request or response fails."""


class MalformedAnalysisError(LLMError):
    """Raised when an LLM response violates the structured analysis schema."""
