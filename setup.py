from setuptools import setup, find_packages

setup(
    name='StomAI',
    version='1.0.0',
    packages=find_packages(
        include=[
            'app',
            'inference',
            'interface',
            'tools',
        ]
    )
)
