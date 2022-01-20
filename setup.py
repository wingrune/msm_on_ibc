import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msm",
    version="0.0.1",
    author="Alexis Thual, Tatiana Zemskova, Thomas Moreau",
    author_email="author@example.com",
    description="Multimodal Surface Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wingrune/msm_on_ibc",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["nibabel", "nilearn", "numpy", "pandas", "sklearn"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
