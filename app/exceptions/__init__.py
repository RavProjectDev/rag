import pkgutil
import inspect
import importlib
from rag.app.exceptions.base import BaseAppException
import rag.app.exceptions as exceptions_pkg

ALL_EXCEPTIONS = []

# Dynamically import and inspect all modules in rag.app.exceptions
for _, module_name, _ in pkgutil.iter_modules(
    exceptions_pkg.__path__, exceptions_pkg.__name__ + "."
):
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseAppException) and obj is not BaseAppException:
            ALL_EXCEPTIONS.append(obj)
