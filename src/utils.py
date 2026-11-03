from colorama import Fore, Back, Style, init
# from functools import wraps
import traceback
import json

import functools
import sys
import inspect

# init(autoreset=True)
def paint_output(color_code, style_code=Style.NORMAL):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Включаем цвет перед запуском функции
            print(f"{style_code}{color_code}", end="")
            try:
                return func(*args, **kwargs)
            finally:
                # Гарантированно сбрасываем цвет после работы функции
                print(Style.RESET_ALL, end="")
        return wrapper
    return decorator

class color:
    def __init__(self, color_code, style_code=Style.NORMAL):
        self.color_code = color_code
        self.style_code = style_code

    def __enter__(self):
        print(f"{self.style_code}{self.color_code}", end="")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(Style.RESET_ALL, end="")

RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BRIGHT = Style.BRIGHT

def disable(func):
    def wrapper(*args, **kwargs):
        # We just print a warning and return None without running the function
        print(f"⚠️ Code skipped: '{func.__name__}' is currently disabled.")
        return None
    return wrapper

def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            caller = inspect.stack()[1].function
            print(f"{Fore.RED}Error in {func.__name__} (called by {caller}): {e} {Style.RESET_ALL}")
            print(f"traceback.format_exc(): {traceback.format_exc()}")
            return 0.0

    return wrapper