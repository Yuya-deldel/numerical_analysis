import Logging
using LinearAlgebra
include("lib.jl")

# A: symmetric positive definite matrix: A = A^t, A > 0

# cf. when B is not singular matrix, B^t B is symmetric positive definite
#     Bx = b -> B^t Bx = B^t b


# modified Cholesky decomposition
# A = LDL^t | L: lower triangle matrix (diagonal component = 1), D: diagonal matrix 
module Cholesky_decomposition_module
export Cholesky_decomposition
export Cholesky_LSE

# A -> (lower diagonal component: L, diagonal component: D)
function Cholesky_dec!(A)
    n = size(A, 1)
    for i = 2 : n
        for j = 1 : i - 1 
            tmp = A[i, j]
            for k = 1 : j - 1
                tmp -= A[i, k] * A[k, k] * A[j, k]
            end
            A[i, j] = tmp / A[j, j]
        end
        for k = 1 : i - 1
            A[i, i] -= A[i, k] * A[i, k] * A[k, k]
        end        
    end
    return A 
end

# A -> (L, D)
function Cholesky_decomposition(A)
    n = size(A, 1)
    C = Cholesky_dec!(copy(A))
    L = zeros(n, n)
    D = zeros(n, n)
    for i = 1 : n, j = 1 : n 
        if i == j 
            L[i, j] = 1
            D[i, j] = C[i, j]
        elseif i > j 
            L[i, j] = C[i, j]
        end
    end
    return L, D
end

# b = Ax = LDL^t x | y = L^t x, b = LDy
function Cholesky_LSE(A, b)
    n = size(A, 1)
    C = Cholesky_dec!(copy(A))
    y = copy(b)
    
    # b = LDy 
    y[1] /= C[1, 1]
    for i = 2 : n
        tmp = y[i]
        for j = 1 : i - 1
            tmp -= C[j, j] * C[i, j] * y[j]
        end
        y[i] = tmp / C[i, i]
    end

    # L^t x = y
    for i = n - 1 : -1 : 1
        for j = i + 1 : n 
            y[i] -= C[j, i] * y[j]
        end
    end
    return y 
end

end

# Gradient descent method 
# minimize f(x) = 1/2 (x,Ax) - (x,b) -> Ax = b 
function Gradient_descent(A, b, init, ϵ, Max_iter=100)
    x = init 
    p = b - A * x 
    r = p
    k = 0
    while true 
        a = dot(p, r) / dot(p, A * p)
        x += a .* p 
        r -= a .* (A * p)
        if norm(r) < ϵ
            break 
        elseif k == Max_iter 
            @warn "convergence is too late: calculation is aborted: in Gradient_descent"
            break
        end

        p = r - (dot(r, A * p) / dot(p, A * p)) .* p 
        k += 1
    end
    println("iterate counter is $k")
    return x 
end

# test 
#=
using .Cholesky_decomposition_module
M1 = [ 2.0 -1.0  0.0  0.0
      -1.0  3.0 -1.0  0.0
       0.0 -1.0  3.0 -1.0
       0.0  0.0 -1.0  2.0]
V1 = [4.0, -10.0, 15.0, -11.0]
println(Cholesky_LSE(M1, V1))
println(Cholesky_decomposition(M1))
=#

M5 = [5 2 0 0 0 0 0 0 0 0
      2 5 2 0 0 0 0 0 0 0
      0 2 5 2 0 0 0 0 0 0
      0 0 2 5 2 0 0 0 0 0 
      0 0 0 2 5 2 0 0 0 0 
      0 0 0 0 2 5 2 0 0 0 
      0 0 0 0 0 2 5 2 0 0
      0 0 0 0 0 0 2 5 2 0
      0 0 0 0 0 0 0 2 5 2
      0 0 0 0 0 0 0 0 2 5]
V5 = [3; 1; 4; 0; 5; -1; 6; -2; 7; -15]
println(Gradient_descent(M5, V5, ones(10), 0.00000001)) 