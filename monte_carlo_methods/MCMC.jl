using Random
using Plots
using Statistics

# S(x): action <=> log likelyhood function
# P(x) = exp(-S(x)) / Z ; Z: unknown, S: known 
function metropolis_method(S, dim; init=zeros(dim), step=8.0*ones(dim), iter=10^4)
    X = init
    x = copy(init)
    action = 0.0
    accept_counter = zeros(dim) 
    for i = 1 : iter 
        for n = 1 : dim  
            x_old = copy(x) 
            action_old = S(x...)
            x[n] += (rand() - 0.5) * step[n]        # random walk
            action = S(x...)
            # metropolis test 
            if exp(action_old - action) > rand()
                accept_counter[n] += 1
            else
                x = x_old
            end
        end
        if i % 10 == 0
            X = hcat(X, x)
        end
    end

    # logging
    rate = accept_counter / iter
    println("acception rate: $(rate)")
    return X
end

# test 
#=
S1(x) = (x * x) / 2.0
Y = metropolis_method(S1, 1, iter=100000)
println(mean(Y))
println(std(Y))
histogram(Y)
=#

#=
S2(x) = -log(exp(-(x - 3.0)^2 / 2.0) + exp(-(x + 3.0)^2 / 2.0))
Y = metropolis_method(S2)
histogram(Y)
=#

#=
S3(x, y) = (x * x + y * y + x * y) / 2.0
Y = metropolis_method(S3, 2, iter=10000)
println(size(Y))
#scatter(Y[1, :], Y[2, :])
plot(Y[1, :])
=#

S4(x, y, z) = 0.5 * x * x + y * y + z * z + x * y + y * z + z * x 
Y = metropolis_method(S4, 3, iter=100000)
scatter(Y[2, :], Y[3, :])