from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

from spacy.strings import StringStore
os.chdir('./doc/core/fast_bow')
extensions = [Extension('fast_bow', ['fast_bow.pyx'], language='c++')]

setup(
    name='fast_bow',
    ext_modules=cythonize(
        extensions, 
        annotate=True, 
        compiler_directives={'language_level' : '3'},
    )
)