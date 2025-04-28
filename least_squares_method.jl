# least squares method 
# data: (x1, y1), ..., (xm, ym) ==< fit a_i >==> g(x) = Σ(a_i * ϕ_i(x)) 
# minimize F(a1, ..., an) = Σ(yk - Σ(a_i * ϕ_i(xk)))^2  <=> Σ(yk * ϕ_j(xk) - Σ(a_i * ϕ_i(xk) * ϕ_j(xk))) = 0

include("Linear_Simultaneous_Equations.jl")
using .LU_decompositon_module: LSE

# least squares polynomial: g(x) = Σ a_j *x^{j-1}
# data = [(x1, y1), ..., (xm, ym)] -> [a1, ..., an]
function LSM_polynomial(data, n)
    m = size(data, 1)
    mat = zeros(n, n)
    for i = 1 : n, j = 1 : n 
        for k = 1 : m 
            (x, y) = data[k]
            mat[i, j] += (x ^ (i - 1)) * (x ^ (j - 1))
        end
    end

    vec = zeros(n)
    for i = 1 : n 
        for k = 1 : m 
            (x, y) = data[k]
            vec[i] += y * x ^ (i - 1)
        end 
    end

    return LSE(mat, vec)
end

# test 
println(LSM_polynomial([(0.0, 2.0); (0.2, 2.12); (0.4, 1.62); (0.6, 2.57); (0.8, 1.53); (1.0, 2.0)], 4))