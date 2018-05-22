# Go to terminal, cd to this file's path and "sudo python setup.py install"
from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()
setup(
   name='SA',
   version='1.0',
   description='Twitter Sentiment Analysis Module',
   author='Group N: Nam Nguyen & Nevine Gouda',
   long_description=long_description,
   packages=['scripts'],  #same as name
   install_requires=['nltk', 'pyspark', 'pytypo', 'psutil'], #external packages as dependencies #pip install psutil

)