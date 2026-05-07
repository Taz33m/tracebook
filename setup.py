import ast
from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def read_version() -> str:
    """Read the package version without importing runtime dependencies."""
    version_file = ROOT / "src" / "tracebook" / "_version.py"
    tree = ast.parse(version_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Unable to find __version__ in src/tracebook/_version.py")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

CORE_REQUIREMENTS = [
    "numba>=0.65.1",
    "numpy>=2.2.6",
    "psutil>=7.2.2",
]

DASHBOARD_REQUIREMENTS = [
    "dash>=4.1.0",
    "plotly>=6.7.0",
    "pandas>=2.3.3",
]

ANALYSIS_REQUIREMENTS = [
    "h5py>=3.16.0",
    "matplotlib>=3.10.9",
    "pandas>=2.3.3",
    "plotly>=6.7.0",
    "pyarrow>=24.0.0",
    "seaborn>=0.13.2",
    "tables>=3.10.1",
]

DEV_REQUIREMENTS = [
    "build>=1.5.0",
    "black>=26.3.1",
    "flake8>=7.3.0",
    "mypy>=2.0.0",
    "pytest>=9.0.3",
    "pytest-benchmark>=5.2.3",
    "pytest-cov>=7.1.0",
]

setup(
    name="tracebook",
    version=read_version(),
    author="Taz33m",
    author_email="tazeemmahashin@gmail.com",
    description=(
        "Latency-focused order book simulation with reproducible benchmarks "
        "and trace-level profiling hooks"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Taz33m/tracebook",
    license="MIT",
    license_files=["LICENSE"],
    keywords=[
        "order-book",
        "matching-engine",
        "market-microstructure",
        "benchmarking",
        "profiling",
        "simulation",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Homepage": "https://github.com/Taz33m/tracebook",
        "Changelog": "https://github.com/Taz33m/tracebook/blob/main/CHANGELOG.md",
        "Continuous Integration": "https://github.com/Taz33m/tracebook/actions/workflows/ci.yml",
        "Documentation": "https://github.com/Taz33m/tracebook/tree/main/docs",
        "Issues": "https://github.com/Taz33m/tracebook/issues",
        "Security": "https://github.com/Taz33m/tracebook/blob/main/SECURITY.md",
        "Source": "https://github.com/Taz33m/tracebook",
    },
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
