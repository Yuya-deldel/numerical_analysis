using Random
using Distributions
using Plots
using Statistics

function Hamiltonian_monte_carlo(S, dS, dim; init_x=zeros(dim), iter=10^4, Nτ=20, step=1.0)
    X = init_x
    x = copy(init_x)
    accept_counter = 0
    for i = 1 : iter 
        x_old = copy(x)
        x, H_init, H_end = Leapfrog!(x, S, dS, dim, Nτ, step)
        # metropolis test
        if exp(H_init - H_end) > rand()
            accept_counter += 1
        else
            x = x_old
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

function Leapfrog!(x, S, dS, dim, Nτ, step)
    # initializing momentum
    distribution = Normal()
    p = rand(distribution, dim)

    H_init = Hamiltonian(S, x, p, dim)

    # Leapfrog
    x += p * step * 0.5
    for _ = 2 : Nτ
        p -= dS(x) * step 
        x += p * step * 0.5
    end

    H_end = Hamiltonian(S, x, p, dim)

    return x, H_init, H_end
end

function Hamiltonian(S, x, p, dim)
    norm_p = 0.0
    for i = 1 : dim 
        norm_p += p[i] ^ 2
    end
    return norm_p * 0.5 + S(x...)
end

# test

S4(x, y, z) = 0.5 * x * x + y * y + z * z + x * y + y * z + z * x 

function dS4(x)
    return [x[1] + x[2] + x[3], 2 * x[2] + x[1] + x[3], 2 * x[3] + x[1] + x[2]]
end

Y = Hamiltonian_monte_carlo(S4, dS4, 3, iter=100000)
scatter(Y[2, :], Y[3, :])
#=
println(Y[1, 1])
println(Y[1, 2])
println(Y[1, 3])
println(Y[2, 1])
println(Y[2, 2])
println(Y[2, 3])
println(Y[3, 1])
=#