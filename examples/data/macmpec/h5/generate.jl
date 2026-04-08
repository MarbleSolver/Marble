using JSON

include(joinpath(@__DIR__, "../parser.jl"))

# Read the LCQP MacMPEC problems from the filtered list
json_path = joinpath(@__DIR__, "../macmpec_lcqp.json")
json_data = JSON.parsefile(json_path)

for key in keys(json_data)
    @info "Processing problem: $key"
    
    nl_path = joinpath(@__DIR__, "../ampl_nl", key * ".nl")
    P = problem_data(nl_path=nl_path, dense=true)
    save_data(data=P, savedir=@__DIR__)
end