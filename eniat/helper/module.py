import types

__ignores__ = ['re', 'os', 'sys', 'types', 'threading']

def module_parser(objects: dict, class_only: bool = False) -> list:
    """
    Categorizes and lists objects contained within a module.

    This function segregates objects into functions, classes, and sub-modules based 
    on their type. It's designed to operate on dictionaries like the one returned 
    by the `globals()` function, which represents the global symbol table of a module.

    Parameters:
    - objects (dict): A dictionary of objects, typically representing the content of a module.
                      Expected to contain keys as object names and values as the object references.
    - class_only (bool, optional): If set to `True`, the function will return only 
                                  the classes. Defaults to `False`, returning functions, 
                                  classes, and sub-modules.

    Returns:
    - list: A list containing names of functions, classes, and sub-modules. The type 
            of objects returned is based on the `class_only` parameter.

    Notes:
    - Functions named 'module_parser' and modules listed in `__ignores__` are excluded 
      from the result.
    - Classes are filtered to ensure they belong to one of the listed modules.

    Examples:
    --------
    >>> module_parser(globals())
    ['function_name1', 'ClassName1', 'sub_module_name1', ...]

    >>> module_parser(globals(), class_only=True)
    ['ClassName1', 'ClassName2', ...]
    """
    
    _functions = []
    _classes = []
    _modules = []

    for k, v in objects.items():
        if isinstance(v, types.FunctionType) and k != 'module_parser':
            _functions.append(k)
        elif isinstance(v, type):
            _classes.append({"class_name": k, "module_name": v.__module__.split('.')[-1]})
        elif isinstance(v, types.ModuleType) and k not in __ignores__:
            _modules.append(k)

    # filter classes
    _classes = [c["class_name"] for c in _classes if c["module_name"] in _modules]
    
    return _classes if class_only else _functions + _classes + _modules
