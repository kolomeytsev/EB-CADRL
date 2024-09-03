from setuptools import setup


setup(
    name='ebcadrl',
    version='0.0.1',
    packages=[
        'rl',
        'configs',
        'rl.policy',
        'rl.utils',
        'simulator',
        'simulator',
        'simulator.policy',
        'simulator.utils',
        'simulator.agents',
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
