#!/usr/bin/python

import tablite
import mkdocs_gen_files
import pkgutil

# We are generating files in the virtual mkdocs_gen_files environment, no real files are created
for submodule in pkgutil.iter_modules(tablite.__path__):
    with mkdocs_gen_files.open("reference/{submodule_name}.md".format(submodule_name=submodule.name), "w") as f:
        print("::: tablite.{submodule_name}".format(submodule_name=submodule.name), file=f)
