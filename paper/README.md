# Soft Search 2022 JCDL Paper

This sub-directory is responsible for generating the 2022 JCDL Soft Search Paper.

---

## Setup

1.  Install [Quarto](https://quarto.org/docs/get-started/).
2.  Install [Just](https://github.com/casey/just#packages).
3.  Run `just setup` to create a new conda environment with partial dependencies.
4.  Run `conda activate councils-in-action` to activate the environment.
    - After activating the environment, install the `soft-search` library
      to finish dependency installation (`pip install -e ../`)
5.  Build!
    - `just build` to build the project to the `_build/` folder.
    - `just watch` to watch this directory and build just the website on file save.

You may run into issues running your first `just build` command. If the issue relates to
`tinytex` or `texlive` then try installing the latest versions: `quarto install tinytex`

### Development Commands

```
just --list
Available recipes:
    build   # build page
    clean   # remove build files
    default # list all available commands
    setup name="soft-search-paper" # create conda env and install partial deps
    watch   # watch file, build, and serve
```
