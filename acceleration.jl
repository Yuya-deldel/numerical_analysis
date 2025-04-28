# acceleration of convergence 
# ex1. first order convergence: |a_n+1 - a| <= c|a_n - a|
# ex2. O(n^{-α}) convergence
# ex3. alternating series 

# Romberg-Richardson method 
# when a_n ~ b_0 + b_1 * λ_1 ^ n + b_2 * λ_2 ^ n + ... asymptotically n -> ∞, and 1 > λ_1 > λ_2 > ... are known
# a_n ==<acceleration>==> c_n ~ b_0 + b_2 * λ_2 + ... (droping λ_1 term)
function Romberg_Richardson(a, λ_1)
    n = size(a, 1)
    c = zeros(n - 1)
    for i = 1 : n - 1
        c[i] = (a[i+1] - λ_1 * a[i]) / (1 - λ_1)
    end
    return c
end

# Aitken method
# when a_n ~ b_0 + b_1 * λ_1 ^ n + b_2 * λ_2 ^ n + ... asymptotically n -> ∞, and 1 > λ_1 > λ_2 > ... , but value of λ_1 is NOT known
# when λ_1 ~ λ_2, Romberg-Richardson and Aitken method may fail.
function Aitken_method(a)
    ϵ = 4.0 * eps(Float64)
    n = size(a, 1)
    c = zeros(n)
    for i = 1 : n - 2
        if abs(a[i+1] - a[i]) < ϵ 
            break 
        end
        c[i] = a[i] - (((a[i+1] - a[i]) ^ 2) / (a[i] - 2.0 * a[i+1] + a[i+2]))
    end
    return c
end

# epsilon method 
# when Romberg-Richardson and Aitken_method fail 
# ϵ_6, ϵ_8, ϵ_10, ... will behave well at n -> ∞
function epsilon_method(a, kMax=6)
    n = size(a, 1)
    ϵ = [zeros(n) for _ = 1 : n]
    for i = 1 : n 
        ϵ[2][i] = a[i]
    end
    for k = 3 : kMax 
        for i = k : n 
            ϵ[k][i] = ϵ[k-2][i-1] + 1.0 / (ϵ[k-1][i] - ϵ[k-1][i-1])
        end
    end
    return ϵ[kMax]
end

# test 
#=
a = ones(20)
for i = 2 : 20 
    a[i] = a[i-1] + ((-1) ^ (i-1)) / (2.0 * i - 1.0)
end
println("Aitken method:")
println(Aitken_method(a))
println("epsilon_method:")
println(epsilon_method(a))
=#
b = ones(20)
for i = 2 : 20 
    b[i] = b[i-1] + 1.0 / (i * i)
end
println(Aitken_method(b))