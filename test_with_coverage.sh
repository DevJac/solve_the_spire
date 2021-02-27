#!/usr/bin/env fish

julia --project --code-coverage=tracefile.info test/runtests.jl
genhtml tracefile.info --quiet --ignore-errors=source --output-directory=coverage 2> /dev/null
open coverage/index.html
