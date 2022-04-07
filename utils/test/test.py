
    

import pytest
from pathlib import Path 

if __name__ == "__main__":
    for p in list(Path.cwd().glob("test_*.py")):
        print(p)
        pytest.main(["-s", f"{p}"])