from setuptools import setup
exec(open('rlx/version.py').read())

setup(name='rlx',
      version=__version__,
      description='python rlx tools',
      url='http://github.com/rramosp/rlx',
      install_requires=['matplotlib','numpy', 'pykalman', 'pandas', 'deepdish'],
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlx', 'rlx.topics', 'rlx.auger'],
      include_package_data=True,
      zip_safe=False)
