# eigenvalues of matrix calculator 
module Eigenvalues 
export power_iteration!
export eigenvalues_QR

include("lib.jl")
include("Linear_Simultaneous_Equations.jl")
using LinearAlgebra
import .LU_decompositon_module: LSE

# power method: calculation of the largest eigenvalue and eigenvector 
# speed of convergence: ~ (|λ1| / |λ2|) ^ k 
function power_iteration!(A, ϵ, iter_max=1000)
    k = 0
    λ = 1.0
    x = ones(size(A, 1))
    x = x ./ norm_2(x)
    while k < iter_max
        v = A * x
        λ = dot(x, v)
        norm = norm_2(v)
        x = v / norm 
        k += 1
        if (norm^2 - λ^2) < ϵ
            break
        end
    end
    return x, λ
end

# QR method 
# Householder transformation: convert matrix to Hessenberg matrix 
function Householder_transformation!(A)
    n = size(A, 1)
    for k = 1 : n - 2
        u = zeros(n)
        for j = k + 1 : n 
            u[j] = A[j, k]
        end
        u[k+1] -= sign(A[k+1, k]) * norm_2(u)
        if norm_2(u) < eps(Float64)
            continue 
        end
        u /= norm_2(u)
        f = A * u 
        g = transpose_and_product(A, u, n, k) 
        γ = dot(u, f)
        f -= γ .* u 
        g -= γ .* u 
        for i = 1 : n, j = 1 : n
            A[i, j] -= 2.0 * (u[i] * g[j] + f[i] * u[j])
        end
    end
    return A 
end

function transpose_and_product(Mat, vec, n, k)
    u = zeros(n)
    for i = 1 : n 
        for j = k+1 : n 
            u[i] += Mat[j, i] * vec[j]
        end
    end
    return u 
end

# QR method: convert Hessenberg matrix to upper triangle matrix
function QR_method!(A, ϵ)
    n = size(A, 1)
    A = Householder_transformation!(A)
    m = n 
    tmp_vec = zeros(n)
    while m > 1 
        if abs(A[m, m-1]) < ϵ 
            m -= 1
            continue 
        end
    
        # origin shift 
        shift = 0.0
        if m < n 
            shift = A[n, n]
            for i = 1 : m
                A[i, i] -= shift
            end
        end

        # QR method 
        Q = zeros(m, m)
        for i = 1 : m 
            Q[i, i] = 1.0
        end

        for i = 1 : m - 1
            r = √(A[i, i] * A[i, i] + A[i+1, i] * A[i+1, i])
            sin_t = 0.0
            cos_t = 0.0 
            if r > eps(Float64) 
                sin_t = A[i+1, i] / r 
                cos_t = A[i, i] / r 
            end
            for j = i+1 : m 
                tmp = A[i, j] * cos_t + A[i+1, j] * sin_t 
                A[i+1, j] = -A[i, j] * sin_t + A[i+1, j] * cos_t
                A[i, j] = tmp
            end
            A[i+1, i] = 0.0 
            A[i, i] = r 
            
            for j = 1 : m 
                tmp = Q[j, i] * cos_t + Q[j, i+1] * sin_t 
                Q[j, i+1] = -Q[j, i] * sin_t + Q[j, i+1] * cos_t 
                Q[j, i] = tmp 
            end
        end
        
        for i = 1 : m 
            for j = 1 : m 
                tmp_vec[j] = A[i, j]
            end
            for j = 1 : m 
                tmp = 0.0
                for k = i : m 
                    tmp += tmp_vec[k] * Q[k, j]
                end
                A[i, j] = tmp 
            end
        end
        for i = 1 : m 
            A[i, i] += shift
        end
    end
    return A 
end

# inverse power method 
# calculation of eigenvector corresponds to eigenvalue
function eigenvector(A, λ, ϵ, iter_max=1000)
    n = size(A, 1)
    λE = zeros(n, n)
    for i = 1 : n 
        λE[i, i] = λ
    end
    D = A - λE

    y = ones(n)
    y = y ./ norm_2(y)
    μ = 0.0
    k = 0
    while k < iter_max 
        v = LSE(D, y)
        μ_prev = μ
        μ = dot(y, v)
#        λ += 1.0 / μ
        y = v ./ norm_2(v)
        k += 1
        if (μ - μ_prev) / μ < ϵ
            break
        end 
    end
    return y
end

# eigenvalues by QR method
function eigenvalues_QR(A, ϵ)
    n = size(A, 1)
    R = QR_method!(copy(A), ϵ)
    eigen = []
    for i = 1 : n 
        push!(eigen, (R[i, i], eigenvector(A, R[i, i], ϵ)))
    end
    return eigen 
end

end

# test 
using .Eigenvalues: eigenvalues_QR
M1 = [16.0 -1.0 1.0 2.0
      2.0 12.0 1.0 -1.0
      1.0 3.0 -24.0 2.0
      4.0 -2.0 1.0 20.0]
V1 = [0.5; 0.5; 0.5; 0.5]
# println(power_iteration!(M1, V1, 0.00000001))
println(eigenvalues_QR(M1, 0.00000001))

M2 = [1.0 1.0 0.0
      0.0 3.0 -4.0
      0.0 1.0 -2.0]
# println(Householder_transformation!(M2))
# println(eigenvalues_QR(M2, 0.00000001))