from setuptools import setup, find_packages

setup(
    name="hiver-ai-support-agent",
    version="1.0.0",
    author="Hiver SDE Intern Candidate",
    description="Production-grade AI Customer Support Agent for @AppleSupport on Twitter",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "pytest",
    ],
)
