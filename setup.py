from distutils.core import setup

setup(name='Machine_Learning: Model Order Reduction and ANN',
  version= '1.0.0',
  description='Model Order Reduction and ANN - Université Libre de Bruxelles',
  author='Giuseppe D Alessio',
  author_email= 'giuseppe.dalessio@ulb.ac.be',
  packages = ['src'],
  scripts = ['my_exec.py'],
  url = 'https://github.com/gdalessi/Machine_Learning',
  download_url = 'https://github.com/gdalessi/Machine_Learning.git',
  keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction', 'model order reduction'])




'''

How to install the package:

$ python setup.py sdist

Then it the ouput you'll have something like:

running sdist
running check
warning: sdist: manifest template 'MANIFEST.in' does not exist (using default file list)

warning: sdist: standard file not found: should have one of README, README.txt

writing manifest file 'MANIFEST'
creating Machine_Learning and Model Order Reduction-1.0.0
creating Machine_Learning and Model Order Reduction-1.0.0/src
making hard links in Machine_Learning and Model Order Reduction-1.0.0...
'my_exec.py' not a regular file -- skipping
hard linking setup.py -> Machine_Learning and Model Order Reduction-1.0.0
hard linking src/ANN.py -> Machine_Learning and Model Order Reduction-1.0.0/src
hard linking src/__init__.py -> Machine_Learning and Model Order Reduction-1.0.0/src
hard linking src/clustering.py -> Machine_Learning and Model Order Reduction-1.0.0/src
hard linking src/model_order_reduction.py -> Machine_Learning and Model Order Reduction-1.0.0/src
hard linking src/utilities.py -> Machine_Learning and Model Order Reduction-1.0.0/src
creating dist
Creating tar archive
removing 'Machine_Learning and Model Order Reduction-1.0.0' (and everything under it)


The resultant distribution will be saved as Machine_Learning and Model Order Reduction-1.0.tar.gz’ inside the ‘dist’ folder.

Extract the ‘zipped’ file. Then execute the setup.py file in the unzipped folder as below


$ cd dist/Machine_Learning and Model Order Reduction-1.0.0
$ python setup.py install

'''