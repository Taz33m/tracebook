from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

CORE_REQUIREMENTS = [
    "numba>=0.58.0",
    "numpy>=1.24.0",
    "psutil>=5.9.0",
]

DASHBOARD_REQUIREMENTS = [
    "dash>=2.12.0",
    "plotly>=5.15.0",
    "pandas>=2.0.0",
]

ANALYSIS_REQUIREMENTS = [
    "h5py>=3.9.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "plotly>=5.15.0",
    "pyarrow>=12.0.0",
    "seaborn>=0.12.0",
    "tables>=3.8.0",
]

DEV_REQUIREMENTS = [
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pytest>=7.4.0",
    "pytest-benchmark>=4.0.0",
    "pytest-cov>=4.1.0",
]

setup(
    name="tracebook",
    version="0.1.0",
    author="Taz33m",
    description="High-performance order book simulator with Numba optimization and magic-trace profiling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Taz33m/tracebook",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "analysis": ANALYSIS_REQUIREMENTS,
        "dashboard": DASHBOARD_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
        "all": sorted(
            set(
                CORE_REQUIREMENTS
                + DASHBOARD_REQUIREMENTS
                + ANALYSIS_REQUIREMENTS
                + DEV_REQUIREMENTS
            )
        ),
    },
    entry_points={
        "console_scripts": [
            "tracebook-sim=tracebook.simulation.simulation_engine:main",
            "tracebook-benchmark=tracebook.benchmarks.runner:main",
            "tracebook-dashboard=tracebook.visualization.dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
