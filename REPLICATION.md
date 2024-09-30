# Replication Guide

Welcome to the Climate Econometrics repository. This guide will help you get started with cloning the repository and running the notebooks on your local machine. Follow the steps below to replicate the analyses.

## Prerequisites

Before you begin, make sure you have the following software installed:

1. **Git**: Version control system for cloning the repository.
    - [Download Git](https://git-scm.com/downloads)
2. **Python**: Programming language required to run the notebooks.
    - [Download Python](https://www.python.org/downloads/)
3. **Jupyter Notebook**: Interactive computing environment for running the notebooks.
    - You can install Jupyter Notebook via pip:
      ```sh
      pip install notebook
      ```

## Step-by-Step Guide

### 1. Cloning the Repository

1. **Create a GitHub Account**: If you don't already have a GitHub account, sign up at [github.com](https://github.com/).

2. **Fork the Repository**: Go to the repository page and click the "Fork" button in the top right corner to create a copy of the repository under your own GitHub account.

3. **Clone the Repository**: Open a terminal or command prompt on your local machine and run the following commands:
    ```sh
    git clone https://github.com/dhruv-pandit/climate-econometrics.git
    cd climate-econometrics
    ```

### 2. Setting Up the Environment

1. **Create a Virtual Environment**: It's recommended to create a virtual environment to manage dependencies.
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

2. **Install Dependencies**: Install the required Python packages using the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```

### 3. Running the Notebooks

1. **Navigate to the Notebook**: Navigate to the directory containing the notebook you want to run.
    ```sh
    cd templates  # or any specific folder like countries/USA/notebooks
    ```

2. **Start Jupyter Notebook**: Launch Jupyter Notebook to open and run the notebook.
    ```sh
    jupyter notebook
    ```

3. **Open the Notebook**: In the Jupyter Notebook interface that opens in your web browser, click on the notebook file (e.g., `template_study_notebook.ipynb`) to open it.

4. **Run the Notebook**: Execute the cells in the notebook by clicking "Run" in the toolbar or pressing `Shift + Enter`.

### Troubleshooting

- **Git Not Installed**: If you encounter an error that `git` is not recognized, ensure that Git is installed and added to your system PATH.
- **Dependencies Issues**: If there are issues with missing dependencies, double-check that all packages listed in `requirements.txt` are installed correctly.
- **Jupyter Not Starting**: If Jupyter Notebook fails to start, ensure it is installed and try running `jupyter notebook` again.

## Additional Resources

- [GitHub Documentation](https://docs.github.com/en)
- [Python Documentation](https://docs.python.org/3/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/)

---

For any further assistance, please open an issue in the repository or contact [dpandit@novaims.unl.pt].

Happy replicating!