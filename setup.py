# # -*- coding: utf-8 -*-
# from setuptools import setup

# packages = \
# ['casino']

# package_data = \
# {'': ['*']}

# install_requires = \
# ['cython>=0.29.35,<0.30.0', 'numba>=0.57.0,<0.58.0', 'numpy>=1.24.3,<2.0.0']

# setup_kwargs = {
#     'name': 'casino',
#     'version': '0.1.0',
#     'description': 'Casino enables fast probabilistic modelling',
#     'long_description': '# Casino Probabilistic Modelling Library\n\nThis is awesome.',
#     'author': 'Adolph Vogel',
#     'author_email': 'ajvogel@gmail.com',
#     'maintainer': 'None',
#     'maintainer_email': 'None',
#     'url': 'None',
#     'packages': packages,
#     'package_data': package_data,
#     'install_requires': install_requires,
#     'python_requires': '>=3.11,<3.12',
# }
# from build import *
# build(setup_kwargs)

# setup(**setup_kwargs)


import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Python's own CFLAGS stop at -O2. -O3 is worth 11-16% here, almost all of it in
# _add()'s shift loops, which only vectorise now that the buffers are declared
# contiguous. -fno-math-errno lets the C compiler treat sqrt/log/exp as pure,
# which it needs to hoist them out of the sampling loops.
#
# -march=native is deliberately opt-in: it bakes in the build machine's
# instruction set, which is right for a local build and wrong for any wheel that
# leaves it. Set RANVAR2_NATIVE=1 to enable it.
if sys.platform == 'win32':
    OPTIMISE = ['/O2']
else:
    OPTIMISE = ['-O3', '-fno-math-errno']

    if os.environ.get('RANVAR2_NATIVE'):
        OPTIMISE.append('-march=native')

extensions = [
    Extension("ranvar2.ranvar", ["ranvar2/ranvar.py"], extra_compile_args=OPTIMISE),
]

setup(
    name="ranvar2",
    ext_modules=cythonize(
        extensions,
        include_path=["ranvar2/", np.get_include()],
        language_level=3,
    ),
)
