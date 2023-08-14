class SilentIO:
    """
    A file-like object that suppresses all write operations. 
    Useful for redirecting and silencing output.
    
    Examples:
    --------
    ```python
    import sys
    sys.stdout = SilentIO()  # Redirect standard output to suppress print statements
    print("This won't be printed to the console.")
    ```
    """
    def __init__(self):
        """Initializes a new instance of the SilentIO class."""
        pass

    def write(self, message: str) -> None:
        """
        Overridden write method that does nothing.
        
        Parameters:
        - message : str
            The message intended to be written. In this implementation, the message will be discarded.
        """
        pass