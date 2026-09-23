project(
  'scikit-clarans',
  'c', 'cython',
  version: '0.3.1',
  license: 'MIT',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
  ],
)

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

# NumPy C API include directory
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_numpy = include_directories(incdir_numpy)

# C compiler flags and macros
c_args = ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']

subdir('clarans')
