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
        "attrs",
        "jsonschema",
        "tqdm",
        # TODO: Remove boto3
        "boto3",
        "requests",
        "aiohttp",
        "pandas",
        "requests-toolbelt",
        "descope",
        "pypandoc-binary",
        "pypdf",
    ],
    # This sorta works for Test-Pypi, but not fully.
    # use_scm_version={"local_scheme": "no-local-version"},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    extras_require={
        # Requirements only needed for development, not for users
        "dev": ["ariadne-codegen"],
    },
    entry_points={
        "console_scripts": [
            "vals = vals.cli.main:cli",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
