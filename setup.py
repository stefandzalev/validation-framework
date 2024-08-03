from setuptools import find_packages, setup
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_project_requirements() -> str:
    with open(f"{PROJECT_ROOT_DIR}/requirements.txt", "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="data-lake-machine-learning",
    version="1.0",
    author="",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_project_requirements(),
)
