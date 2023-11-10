from setuptools import setup, find_packages

setup(
  name = 'gateloop-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.19',
  license='MIT',
  description = 'GateLoop Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/gateloop-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'gated linear attention'
  ],
  install_requires=[
    'einops>=0.7.0',
    'rotary-embedding-torch',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
