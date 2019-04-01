from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='python-stock-tool',
      version='0.1',
      description='python-stock-tool',
      long_description=readme(),
      classifiers=[],
      keywords='stock tool',
      url='https://github.com/tabano86/python-stock-analysis/',
      author='Anthony Tabano',
      author_email='tabano86@gmail.com',
      packages=['src'],
      install_requires=[
          'matplotlib', 'numpy', 'pandas', 'keras', 'sklearn'
      ],
      entry_points={
          'console_scripts': ['src=console:main'],
      },
      include_package_data=True,
      zip_safe=False)
