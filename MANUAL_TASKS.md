# Manual Tasks Required

- [ ] Manual fix needed in `01_logistic_regression_manuscript.ipynb`: oding=ioargs.encoding,
    930             errors=errors,
    931             newline="",
    932         )
    933     else:
    934         # Binary mode
    935         handle = open(handle, ioargs.mode)

FileNotFoundError: [Errno 2] No such file or directory: 'manuscript_authenticity_data.csv'


- [ ] Manual fix needed in `02_logistic_regression_silicon.ipynb`: g=ioargs.encoding,
    930             errors=errors,
    931             newline="",
    932         )
    933     else:
    934         # Binary mode
    935         handle = open(handle, ioargs.mode)

FileNotFoundError: [Errno 2] No such file or directory: 'path/to/silicon_timing_test_data.csv'


