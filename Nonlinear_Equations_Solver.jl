# Nonlinear Equations solver 
import Logging
include("lib.jl")
include("Linear_Simultaneous_Equations.jl")
using .LU_decompositon_module: LSE

# bisection method 
function bisection(func, interval, ϵ)
    a, b = interval
    c = (a + b) / 2.0
    while (b - a) >= ϵ
        if func(a) * func(c) < 0.0
            b = c 
        else
            a = c 
        end
        c = (a + b) / 2.0
    end
    return c 
end

function NLE_bisection(func, interval, ϵ, h=0.1)
    min, max = interval
    machine_error = 8 * eps(Float64)
    if abs(func(min)) < machine_error 
        throw("invalid interval: f(lower_boundary) ≈ 0")
    elseif abs(func(max)) < machine_error
        throw("invalid interval: f(upper_boundary) ≈ 0")
    elseif min > max 
        tmp = min 
        min = max 
        max = tmp
    end

    (x, y) = (min, min + h)
    roots = []
    while y < max + h
        if abs(func(y)) < machine_error 
            y += h 
        else
            if func(x) * func(y) < 0.0 
                push!(roots, bisection(func, (x, y), ϵ))
            end
            x = y 
            y += h 
        end
    end
    return roots
end

# Newtonian method 
function Newtonian_method(func, derivative_func, initial_value, ϵ, Max_iter=10)
    x = initial_value
    n = 0
    d = Inf64
    while abs(d) > ϵ
        if n > Max_iter 
            @warn "solution could not be found: choose initial_value and ϵ again: in Newtonian_method"
            x = initial_value
            break 
        end
        d = - func(x) / derivative_func(x)
        x += d 
        n += 1
    end
    return x 
end

# secant method 
function secant_method(func, initial_value, ϵ, Max_iter=10) 
    x = initial_value
    n = 0
    d = Inf64
    h = ∛ϵ / 2.0
    while abs(d) > ϵ
        if n > Max_iter 
            @warn "solution could not be found: choose initial_value and ϵ again: in Newtonian_method"
            x = initial_value
            break 
        end
        d = - func(x) * 2.0 * h / (func(x+h) - func(x-h))
        x += d 
        n += 1
    end
    return x 
end

# for simultaneous equations 

function jacobian(func, x, h)
    n = size(x, 1)
    J = ones(n, n)
    for j = 1 : n 
        d = zeros(n)
        d[j] = h
        func_plus = func(x + d)
        func_minus = func(x - d)
        for i = 1 : n 
            J[i, j] = (func_plus[i] - func_minus[i]) / (2.0 * h)
        end
    end
    return J  
end

function Nonlinear_Simultaneous_Equations(func, initial_value, ϵ, Max_iter=20)
    x = initial_value
    n = 0
    h = ∛ϵ / 2.0
    d = - func(x)
    while norm(d) > ϵ
        if n > Max_iter 
            @warn "solution could not be found: choose initial_value and ϵ again: in Nonlinear_Simultaneous_Equations"
            x = initial_value
            break 
        end
        d = LSE(jacobian(func, x, h), -func(x))
        x += d 
        n += 1
    end
    return x 
end

# test
#=
print("test bisection 1: ")
f(x) = x * x - 1.0
println(abs(1.0 - bisection(f, (0.66, 3.1415), 0.0001)) < 0.0001)

print("test NLE_bisection 1: ")
println(NLE_bisection(f, (-1.5, 1.5), 0.0001, 0.1))

print("test NLE_bisection 2: ")
g(x) = x^5 - 5x^3 + 4x
dg(x) = 5x^4 - 15x^2 + 4
println(NLE_bisection(g, (-5.0, 5.0), 0.0000001, 0.1))

print("test Newtonian_method 1: ")
H(x) = x - cos(x)
dH(x) = 1 + sin(x)
println(Newtonian_method(H, dH, 3.0, 0.00000001))
println(Newtonian_method(g, dg, 1.2, 0.00000001))

print("test secant_method 1: ")
println(secant_method(H, 3.0, 0.00000001))
println(secant_method(g, 1.2, 0.00000001))
=#

function F(X)
    return [X[1]^2 + X[2]^2 + X[1]*X[3] - X[1] - X[2] - 1;
            X[1]^3 + X[3]^3 + 3*X[1]^2 - X[3]^2 + 2*X[2]*X[3] - 2*X[3] - 4;
            3*X[1]*X[2] + 2*X[1]*X[3] + 4*X[2]*X[3] - 3*X[2] - 4*X[3] - 2*X[1]]
end

println(Nonlinear_Simultaneous_Equations(F, [1.1, 1.2, 1.3], 0.00001))
println(Nonlinear_Simultaneous_Equations(F, [1.2, 1.2, 1.2], 0.00001))