from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shield-ai-ml",
    version="2.0.0",
    author="SHIELD Team",
    author_email="contact@shield-safety.com",
    description="AI/ML backend for Women's Safety Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shield-team/ai-ml-backend",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "shield-ai=scripts.start_server:main",
            "shield-train=scripts.train_models:main",
        ],
    },
    include_package_data=True,
)