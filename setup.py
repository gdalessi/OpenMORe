from distutils.core import setup

setup(name='Machine_Learning',
  version= '0.0.1',
  description='Model Order Reduction and Artificial Neural Network functions',
  author='Giuseppe D Alessio',
  package_dir={'': ['src']},
  py_modules=['clustering', 'ANN', 'utilities', 'model_order_reduction'],
  requires = ['numpy', 'keras', 'matplotlib', 'matplotlib.pyplot', 'os'],
  url = 'https://github.com/gdalessi/Machine_Learning',
  download_url = 'https://github.com/gdalessi/Machine_Learning.git',
  keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction', 'model order reduction']
 )