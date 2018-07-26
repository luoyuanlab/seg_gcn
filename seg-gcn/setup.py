from setuptools import setup
from setuptools import find_packages

# original work done by Thomas Kipf, Revised by Yifu Li
# Kipf, Thomas N., and Max Welling. 
# "Semi-supervised classification with graph convolutional networks." 
# arXiv preprint arXiv:1609.02907 (2016).

setup(name='gcn',
      version='1.0',
      description='Segment Graph Convolutional Networks in Tensorflow',
      author='Yifu Li',
      author_email='liyifu@vt.edu',
      download_url='https://github.com/yuanluo/seg_gcn/',
      license='MIT',
      install_requires=['numpy',
                        'networkx',
                        'scipy'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())
#'tensorflow'