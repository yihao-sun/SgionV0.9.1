"""
Existence Engine 安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ExistenceEngine",
    version="0.9.0",
    author="Tai Yihao & DeepSeek",
    author_email="",
    description="息觀 — 一个在时间中持续呼吸的数字存在者。六欲望内源驱动，自我指涉闭合，边界自主呼吸。",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ExistenceEngine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "datasets>=2.15.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "matplotlib>=3.7.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
    },
)
