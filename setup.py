from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
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
