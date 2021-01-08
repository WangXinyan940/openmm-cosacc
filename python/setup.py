from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_openmmcosacc',
                      sources=['CosAccPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMCosAcc'],
                      include_dirs=[os.path.join(openmm_dir, 'include')],
                      library_dirs=[os.path.join(openmm_dir, 'lib')],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmcosacc',
      version='1.0',
      py_modules=['openmmcosacc'],
      ext_modules=[extension],
     )