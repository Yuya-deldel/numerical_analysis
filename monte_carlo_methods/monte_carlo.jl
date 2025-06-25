using Random
using Plots
using Statistics

# f(x): integrand 
# χ(x): characteristic function of integral domain: in [0.0, 1.0]^dim
function monte_carlo(f, χ, dim; iter=10^6)
    sum = 0.0
    for _ = 1 : iter 
        x = rand(dim) 
        if χ(x...)
            sum += f(x...)
        end
    end
    return sum / iter 
end

# test 
f(x) = sqrt(1.0 - x * x)
integral(x) = (x >= 0.0) && (x <= 1.0)
println(monte_carlo(f, integral, 1) * 4)

