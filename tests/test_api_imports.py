import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_backend_imports_without_the_packages_the_slim_image_omits():
    """requirements-api.txt leaves out Streamlit, seaborn and TensorBoard. If the backend
    (or anything it imports) ever needs one of them, the Docker image would break."""
    code = (
        "import sys\n"
        "for m in ('streamlit', 'seaborn', 'tensorboard'):\n"
        "    sys.modules[m] = None  # makes `import m` raise ImportError\n"
        "import backend.main\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout