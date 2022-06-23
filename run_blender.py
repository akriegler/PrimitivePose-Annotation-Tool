"""
    FUNCTION: scene_utils.py
----------------------------------------------------------------------------------------------------------
    DESCRIPTION:
        Used to programmatically launch Blender in a PyCharm (IDE) window
----------------------------------------------------------------------------------------------------------
    RESPONSIBILITY: A. Kriegler (Andreas.Kriegler@ait.ac.at)
    VERSIONs:       python 3.7
    CREATION DATE:  2021/05/10
----------------------------------------------------------------------------------------------------------
    AIT - Austrian Institute of Technology
----------------------------------------------------------------------------------------------------------
"""
import os
import subprocess


def run_blender():
    if os.name == 'nt':
        blender = os.path.join(r"""C:\global_path\blender\blender.exe""")
        subprocess.run([blender, r"""C:\global_path\PrimitivePose\demo\blender\example.blend"""])
    elif os.name == 'posix':
        blender = os.path.join('/home/krieglera/Blender/blender-3.0.1-linux-x64/blender')
        subprocess.run([blender, './iros_tabletop.blend'])


if __name__ == "__main__":
    run_blender()
