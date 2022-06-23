"""
    FUNCTION: pip_install.py
----------------------------------------------------------------------------------------------------------
    DESCRIPTION:
        Used to programmatically install pip packages with the Blender python
----------------------------------------------------------------------------------------------------------
    RESPONSIBILITY: A. Kriegler (Andreas.Kriegler@ait.ac.at)
    VERSIONs:       python 3.7
    CREATION DATE:  2021/05/10
----------------------------------------------------------------------------------------------------------
    AIT - Austrian Institute of Technology
----------------------------------------------------------------------------------------------------------
"""

import os
import sys
import subprocess

python_exe = None

if os.name == 'nt':
    python_exe = sys.executable
elif os.name == 'posix':
    python_exe = os.path.join(sys.prefix, 'bin', 'python3.9')
else:
    print('Unknown platform/operating system.')
    raise NotImplementedError

subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.call([python_exe, "-m", "pip", "install", "opencv-python"])
subprocess.call([python_exe, "-m", "pip", "install", "scipy"])
subprocess.call([python_exe, "-m", "pip", "install", "easydict"])
subprocess.call([python_exe, "-m", "pip", "install", "pyyaml"])
subprocess.call([python_exe, "-m", "pip", "install", "pandas"])
subprocess.call([python_exe, "-m", "pip", "install", "tqdm"])
