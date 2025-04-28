# linear simultaneous equation solver
# LU decomposition method
module LU_decompositon_module
export LSE
const ϵ = eps(Float64) * 4

# LU_decomposition: A -> (P, L) s.t. PA = LU
# L: lower triangle matrix, U: upper triangle matrix, P: permutation
function LU_decomposition!(A)
    n = size(A, 1)
    P = [i for i = 1 : n]
    for k = 1 : n - 1        # column
        # pivot selection
        a_max = abs(A[k, k])
        pivot = k 
        for i = k + 1 : n 
            if abs(A[i, k]) > a_max 
                a_max = abs(A[i, k])
                pivot = i 
            end
        end
        if a_max < ϵ
            error("matrix is singular: det(A) ≈ 0")
        end

        # pivot permutation
        P[k] = pivot
        if pivot != k 
            for j = k : n
                tmp = A[k, j] 
                A[k, j] = A[pivot, j]
                A[pivot, j] = tmp
            end
        end

        # forward elimination : ~ O(n^3)
        for i = k + 1 : n       # row below diagonal component
            A[i, k] = A[i, k] / A[k, k]       
            for j = k + 1 : n 
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return P, A 
end

function LSE_solver!((P, L), y)
    n = size(L, 1)

    # calculating y : Ly = Pb
    for k = 1 : n - 1
        # pivot permutation
        tmp = y[k]
        y[k] = y[P[k]]
        y[P[k]] = tmp

        for i = k + 1 : n
            y[i] -= L[i, k] * y[k]
        end
    end

    # calculating x : Ux = y
    for k = n : -1 : 1
        tmp = y[k]
        for j = k + 1 : n
            tmp -= L[k, j] * y[j]
        end
        y[k] = tmp / L[k, k]
    end
    return y 
end

function scaling!(A, b)
    n = size(A, 1)
    for i = 1 : n 
        a_max = 0
        for j = 1 : n 
            if a_max < abs(A[i, j])
                a_max = abs(A[i, j])
            end
        end
        for j = 1 : n 
            A[i, j] /= a_max
        end
        b[i] /= a_max
    end
    return A, b  
end

function LSE(A, b)
    (A_scaled, b_scaled) = scaling!(copy(A), copy(b)) 
    LSE_solver!(LU_decomposition!(A_scaled), b_scaled)
end

end

# Gaussian elimination method 
module Gaussian_elimination_module
export LSE
const ϵ = 2 * eps(Float64)

# Ax = b -> x = A^{-1}b
function LSE(A, b)
    LSE!(copy(A), copy(b))
end

function LSE!(A, b)
    if size(A, 2) != size(b, 1) 
        error("Matrix size error: size(A, 2) != size(b): in LSE()")
    elseif size(A, 2) != size(A, 1)
        error("LSE() is only defined to square matrix")
    end 

    n = size(A, 1)
    for k = 1 : n - 1           # column
        # pivot selection
        a_max = abs(A[k, k])
        pivot = k 
        for i = k + 1 : n 
            tmp = abs(A[i, k])
            if tmp > a_max 
                a_max = tmp
                pivot = i
            end 
        end
        if a_max < ϵ
            error("Matrix is singular: in LSE()")
        end

        # pivot replacement
        if pivot != k 
            for j = k : n 
                tmp = A[k, j]
                A[k, j] = A[pivot, j]
                A[pivot, j] = tmp
            end 
            tmp = b[k]
            b[k] = b[pivot]
            b[pivot] = tmp
        end

        # forward elimination : ~ O(n^3)
        for i = k + 1 : n       # row below diagonal component
            α = - A[i, k] / A[k, k]
            for j = k + 1 : n 
                A[i, j] += α * A[k, j]
            end 
            b[i] += α * b[k]
        end 
    end

    # backward substitution
    for k = n : -1 : 1
        tmp = b[k]
        for j = k + 1 : n
            tmp -= A[k, j] * b[j]
        end 
        b[k] = tmp / A[k, k]
    end 
    return b 
end

end

# iterative methods for sparse matrix
# Ax = b -> x = Mx + Nb | M: iterate matrix
# A = L + D + U | L: lower matrix, D: diagonal matrix, U: upper matrix
module LSE_for_sparse_matrix_module 
import Logging
include("lib.jl")

export LSE_Jacobian 
export LSE_Gauss_Seidel
export SOR

# Jacobian method: M = - D^{-1} (L + U), N = D^{-1}
function LSE_Jacobian(A, b, init, ϵ, Max_iter=100)
    n = size(init, 1)
    k = 0
    x = init + ϵ .* ones(n)
    x_next = init
    while norm(x_next - x) > ϵ 
        if k == Max_iter
            @warn "convergence is too late: calculation is aborted: in LSE_Jacobian"
            break 
        end

        for i = 1 : n 
            x[i] = x_next[i]
        end
        for i = 1 : n
            x_next[i] = b[i] 
            for j = 1 : n 
                if j != i 
                    x_next[i] -= A[i, j] * x[j]
                end
            end
            x_next[i] /= A[i, i]
        end
        k += 1 
    end
    println("iterate counter is $k")
    return x_next
end

# Gauss_Seidel method: M = - (D + L)^{-1} U, N = (D + L)^{-1}
function LSE_Gauss_Seidel(A, b, init, ϵ, Max_iter=100)
    n = size(init, 1)
    k = 0
    x = init + ϵ .* ones(n)
    x_next = init
    while norm(x_next - x) > ϵ 
        if k == Max_iter
            @warn "convergence is too late: calculation is aborted: in LSE_Gauss_Seidel"
            break 
        end

        for i = 1 : n 
            x[i] = x_next[i]
        end
        for i = 1 : n
            x_next[i] = b[i] 
            for j = 1 : i - 1
                x_next[i] -= A[i, j] * x_next[j]
            end
            for j = i + 1 : n 
                x_next[i] -= A[i, j] * x[j]
            end
            x_next[i] /= A[i, i]
        end
        k += 1 
    end
    println("iterate counter is $k")
    return x_next
end

# SOR method: M = - (D + ωL)^{-1} ((1-ω)D - ωU), N = ω(D + ωL)^{-1}
function SOR(A, b, init, ω, ϵ, Max_iter=100)
    n = size(init, 1)
    k = 0
    x = init + ϵ .* ones(n)
    x_next = init
    ξ = init 
    while norm(x_next - x) > ϵ 
        if k == Max_iter
            @warn "convergence is too late: calculation is aborted: in SOR"
            break 
        end

        for i = 1 : n 
            x[i] = x_next[i]
        end
        for i = 1 : n
            ξ[i] = b[i] 
            for j = 1 : i - 1
                ξ[i] -= A[i, j] * x_next[j]
            end
            for j = i + 1 : n 
                ξ[i] -= A[i, j] * x[j]
            end
            ξ[i] /= A[i, i]
            x_next[i] = x[i] + ω * (ξ[i] - x[i])
        end
        k += 1 
    end
    println("iterate counter is $k")
    return x_next
end

end

# test

using .LSE_for_sparse_matrix_module
#=
M1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
V1 = [1.0, 2.0, 3.0]
println(LSE_Jacobian(M1, V1, [1.0, 1.0, 1.0], 0.0001) ≈ [1.0, 2.0, 3.0])

M2 = [1.0 2.0; 3.0 4.0]
V21 = [1.0, 0.0]
V22 = [0.0, 1.0]
println(LSE_Jacobian(M2, V21, [1.0, 1.0], 0.0001)) #≈ [-2.0, 1.5])
println(LSE_Jacobian(M2, V22, [1.0, 1.0], 0.0001)) #≈ [1.0, -0.5])

M3 = [1.0 2.0 1.0 1.0
      4.0 5.0 -2.0 4.0
      4.0 3.0 -3.0 1.0
      2.0 1.0 1.0 3.0]
V3 = [-1.0, -7.0, -12.0, 2.0]
println(LSE_Jacobian(M3, V3, [1.0, 1.0, 1.0, 1.0], 0.0001)) #≈ [-2.0, -1.0, 1.0, 2.0])

M4 = [ 2.0  4.0  1.0 -3.0
      -1.0 -2.0  2.0  4.0
       4.0  2.0 -3.0  5.0
       5.0 -4.0 -3.0  1.0]
V4 = [0.0, 10.0, 2.0, 6.0]
println(LSE_Jacobian(M4, V4, [1.0, 1.0, 1.0, 1.0], 0.0001)) #≈ [2.0, -1.0, 3.0, 1.0])

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
println(SOR(M5, V5, zeros(10), 1.2, 0.00000001)) 
=#