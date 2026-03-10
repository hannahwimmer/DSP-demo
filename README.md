# Some notes on the repo

This repo supports the course 'Digital Signal Processing' and provides a short demo on
how I expect you to structure your home assignments in a meaningful way.

## Package management
I want you to use [`uv`](https://docs.astral.sh/uv/) for package management. To generate
a `.venv`-folder for the first time, or update the dependencies, run `uv sync` in the 
terminal. Add novel packages via `uv add <package_name>`, remove them again if they're 
not needed any longer via `uv remove <package_name>`. To activate the virtual environment,
run `source .venv/bin/activate`. Nothing new under the sun; I think you already used this
tool with Mr. Raab.

## Streamlit
To showcase, document, and present your work done on the home assignments, you're gonna
use [Streamlit](https://docs.streamlit.io/). It's a prototyping framework for **fast**
generation of user interfaces, which makes it a prime choice for interactively displaying
your results, allowing for real-time adaptation of parameters and updated visualization, 
etc. I already added streamlit to the `pyproject.toml` - after running `uv sync`, you 
can use the command `uv run streamlit run Home.py` to run the streamlit demo. It's set
up in a [`multipage setting`](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app),
meaning `Home.py` acts as an entry point (a 'Home page'), and additional files  
in the `pages` folder are accessed through a sidebar. 

Have fun with DSP!


