from setuptools import setup

setup(
    name="signal_simulate",
    packages=[
        "signal_simulate",
        "signal_simulate.simulate",
        "signal_simulate.util",
        "signal_simulate.HPR"
    ],
    install_requires=[
        "numpy",
    ],
    author="zmx",
)
