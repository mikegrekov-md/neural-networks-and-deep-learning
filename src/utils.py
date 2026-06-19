from colorama import Fore, Back, Style, init
from functools import wraps
# init(autoreset=True)
def paint_output(color_code, style_code=Style.NORMAL):
    def decorator(func):
        @wraps(func)
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