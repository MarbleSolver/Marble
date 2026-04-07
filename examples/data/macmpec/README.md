# MacMPEC Benchmark

39 of the MacMPEC benchmark problems fall under the category of LCQP. The following scripts are used to work with the MacMPEC dataset:

- `parser.jl` parses the AMPL `.nl` files and converts each problem into the standard data matrices expected by our solver
- `mat/generate.jl` saves each problem's Marble data matrices to a `.mat` file which stores sparse data matrices which can be loaded in Julia or Python