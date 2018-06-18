from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        packages=find_packages(),
        name='roboshifter',
        version='0.0.0',
        install_requires=[
            'numpy==1.14.3',
            'pandas==0.22.0',
            'tqdm==4.23.0',
            'scikit-learn==0.19.1',
            'scipy==1.1.0',
            'rpy2==2.8.6',
            'simplejson==3.15.0'],
        include_package_data=True)


# Achtung! Other requirements:
#   1) ROOT
#   2) R
#   3) Monet
