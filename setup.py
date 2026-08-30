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


import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("ranvar2.ranvar", ["ranvar2/ranvar.py"]),
    # Extension("ranvar2.vm", ["ranvar2/vm.py"]),
    # Extension("ranvar2.digest", ["ranvar2/digest.py"], include_dirs=[np.get_include()]),
    # Extension(
    #     "ranvar2.cdigest",
    #     sources=["ranvar2/cdigest.pyx"],
    #     include_dirs=[np.get_include(), "ranvar2/cpp"],
    #     language="c++",
    #     extra_compile_args=["-std=c++17"],
    # ),
]

setup(
    name="ranvar2",
    ext_modules=cythonize(
        extensions,
        include_path=["ranvar2/", np.get_include()],
        force=True,
        annotate=True,
    ),
)
