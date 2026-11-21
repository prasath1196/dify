class DatasourceNodeError(ValueError):
    """Base exception for datasource node errors."""

    pass


class DatasourceParameterError(DatasourceNodeError):
    """Exception raised for errors in datasource parameters."""

    pass
