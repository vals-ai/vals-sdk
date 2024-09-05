class ValsException(Exception):
    """
    An exception returned when there is an error querying the Vals SDK.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
