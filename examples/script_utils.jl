module MicrogradExampleScriptUtils

using JSON3

export option_bool,
    option_float,
    option_int,
    option_int_list,
    option_string,
    option_string_list,
    parse_int_list,
    parse_options,
    parse_string_list,
    write_json_output

function parse_options(args)
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if !startswith(arg, "--")
            i += 1
            continue
        end
        parts = split(arg[3:end], "="; limit=2)
        key = parts[1]
        if length(parts) == 2
            options[key] = parts[2]
        elseif i < length(args) && !startswith(args[i + 1], "--")
            options[key] = args[i + 1]
            i += 1
        else
            options[key] = "true"
        end
        i += 1
    end
    return options
end

function parse_int_list(value::AbstractString)
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

function parse_string_list(value::AbstractString)
    return [strip(part) for part in split(value, ",") if !isempty(strip(part))]
end

option_int(options, key, default) = haskey(options, key) ? parse(Int, options[key]) : default
option_float(options, key, default) = haskey(options, key) ? parse(Float64, options[key]) : default
option_string(options, key, default) = haskey(options, key) ? options[key] : default
option_int_list(options, key, default) = haskey(options, key) ? parse_int_list(options[key]) : default
option_string_list(options, key, default) = haskey(options, key) ? parse_string_list(options[key]) : default
option_bool(options, key, default=false) = haskey(options, key) ? lowercase(options[key]) ∉ ("false", "0", "no") : default

function write_json_output(output::Dict{String,Any}, out_path::AbstractString)
    mkpath(dirname(out_path))
    tmp_path = out_path * ".tmp"
    open(tmp_path, "w") do f
        JSON3.pretty(f, output)
    end
    mv(tmp_path, out_path; force=true)
end

end
