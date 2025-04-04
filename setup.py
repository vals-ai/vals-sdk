from setuptools import find_packages, setup

setup(
    name="valsai",
    author="Langston Nashold, Rayan Krishnan",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["jsonschemas/*"]},
    install_requires=[
        "Click",
        "gql",
        "tqdm",
        "requests",
        "pandas",
        "descope",
        "pydantic",
        "httpx",
        "tabulate",
        "inspect-ai",
        "aiohttp",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    extras_require={
        # Requirements only needed for development, not for users
        "dev": ["ariadne-codegen"],
        # Requirements for parsing utilities we provide
        "parsing": ["pypdf", "pypandoc-binary"],
    },
    entry_points={
        "console_scripts": [
            "vals = vals.cli.main:cli",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
