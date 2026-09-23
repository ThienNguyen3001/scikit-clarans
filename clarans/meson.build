# Pure Python sources
python_sources = [
  '__init__.py',
  '_clarans.py',
  '_fast_clarans.py',
  '_initialization.py',
  'utils.py',
]

py.install_sources(
  python_sources,
  subdir: 'clarans'
)

# Package data & type annotations
py.install_sources(
  ['py.typed', '_core.pyi', '_core.pyx'],
  subdir: 'clarans'
)

# Cython extension module
py.extension_module(
  '_core',
  '_core.pyx',
  include_directories: inc_numpy,
  c_args: c_args,
  install: true,
  subdir: 'clarans'
)

subdir('tests')
