from setuptools import setup
setup(
    name="freeferm",
    packages=["freeferm"],
    install_requires=["numpy"],
    version="0.0.1",
    url="https://github.com/sonnerm/freeferm",
    author="Michael Sonner",
    author_email="sonnerm@gmail.com",
    description="Toolkit for performing numerical calculations involving free fermion models",
    extras_require={"sparse linalg":["scipy"]},
)
