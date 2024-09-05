from setuptools import find_packages, setup

setup(
    name="valsai",
    version="0.0.44",
    author="Langston Nashold, Rayan Krishnan",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["jsonschemas/*"]},
    install_requires=[
        "Click",
        "gql",
        "attrs",
        "jsonschema",
        "tqdm",
        "boto3",
        "requests",
        "aiohttp",
        "pandas",
        "requests-toolbelt",
        "descope",
        "pypandoc",
        "pypdf2",
    ],
    entry_points={
        "console_scripts": [
            "vals = vals.cli.main:cli",
        ],
    },
    url="https://pypi.org/project/valsai/",
)
