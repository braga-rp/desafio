from setuptools import setup, find_packages

setup(
    name="desafio kognita",
    version='1.0.0',
    packages=find_packages(),
    url='',
    license='',
    author='Pedro Rangel Braga',
    author_email='pedro.rangel.braga@gmail.com',
    description='',
    setup_requires=['wheel', 'twine'],
    install_requires=[
        "click",
        "smart-open[all]",
        "pandas",
        "seaborn",
        "matplotlib",
        "numpy",
        "sklearn",
        "requests",
        "Flask",
        "Flask-JSON==0.3.4"

    ],
    entry_points={
        'console_scripts': [
            "run_info=desafio.exploratory_data_analysis.info:main",
            "run_stats=desafio.exploratory_data_analysis.stats:main",
            "run_random_forest=desafio.model.random_forest_model.main:main",
            "run_logistic_regression=desafio.model.logistic_regression_model.main:main",
            "run_prediction=desafio.model.prediction.prediction:main",
            "run_api=desafio.api.main:main"
        ]
    }
)
