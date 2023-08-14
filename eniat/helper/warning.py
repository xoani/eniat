import warnings


def deprecated_warning(dep_func: str, alt_func: str, future: bool = False) -> None:
    """
    Issue a warning about a function being deprecated.

    Parameters:
    - dep_func (str): Name of the deprecated function.
    - alt_func (str): Name of the alternative function to use instead.
    - future (bool): If True, it's a pending deprecation (i.e., will be deprecated in the future). 
                     Otherwise, it's already deprecated. Default is False.

    Note:
    - The warning will point out the caller's location for clearer reference.
    """

    if future:
        category = PendingDeprecationWarning
        tense = 'will be'
    else:
        category = DeprecationWarning
        tense = 'is'

    # Enable default behavior for warning
    warnings.simplefilter("default")
    # Generate the warning message
    warnings.warn(f"{dep_func} {tense} deprecated. Please use {alt_func} instead.", category, stacklevel=2)
    # Restore warning filter
    warnings.simplefilter("ignore")


def user_warning(message: str) -> None:
    """
    Issue a user warning.

    Parameters:
    - message (str): The warning message to display to the user.

    Note:
    - The warning will point out the caller's location for clearer reference.
    """

    # Enable default behavior for warning
    warnings.simplefilter("default")
    # Generate the warning message
    warnings.warn(message, UserWarning, stacklevel=2)
    # Restore warning filter
    warnings.simplefilter("ignore")
