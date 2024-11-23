import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class CustomInstallCommand(install):
    """Customized setuptools install command - runs a shell script before installation."""
    def run(self):
        # Ensure the shell script is executable
        subprocess.check_call(['chmod', '+x', 'install-fluxpype-macos.sh'])
        # Execute the shell script and capture output
        result = subprocess.run(['./install-fluxpype-macos.sh'], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        # Proceed with the installation
        install.run(self)

# Check if `src` directory exists and use it, otherwise use current directory
package_dir = {'': 'src'} if os.path.isdir('src') else {'': '.'}
packages = find_packages(where=package_dir[''])

setup(
    name='fluxpype',
    version='0.1.2',
    description='A basic Python project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gillyspace27/fluxpype',
    packages=packages,
    package_dir=package_dir,
    python_requires='>=3.8',
    install_requires=[
        'matplotlib',
        'sunpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
