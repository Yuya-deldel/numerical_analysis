# interpolation

# Lagrange interpolation
# data: [(x1, f1), ..., (xm, fm)] -> polynomial P(x)
function Lagrangian_Polynomial(data, x)
    n = size(data, 1)
    answer = 0.0
    for i = 1 : n
        (z, f) = data[i]
        tmp = 1.0
        for k = 1 : n 
            if k != i 
                (a, b) = data[k]
                tmp *= (x - a) / (z - a)
            end
        end
        answer += f * tmp
    end
    return answer
end

# Newtonian interpolation
# construct P_n(x) successively
function Newtonian_Polynomial(data, x)
    n = size(data, 1)
    Delta = zeros(n, n)
    X = zeros(n)
    for i = 1 : n 
        (z, f) = data[i]
        Delta[i, 1] = f
        X[i] = z
    end
    
    for j = 2 : n 
        for i = 1 : n - j + 1
            Delta[i, j] = (Delta[i+1, j-1] - Delta[i, j-1]) / (X[i+j-1] - X[i])
        end
    end

    pn = 0.0
    for i = 1 : n 
        tmp = Delta[1, i]
        for j = 1 : i - 1 
            tmp *= x - X[j]
        end
        pn += tmp
    end
    return pn 
end

# test 

using Plots
data = [(0.0, 2.0); (0.2, 2.1); (0.4, 1.6); (0.6, 2.6); (0.8, 1.5); (1.2, 2.7); (1.4, 0.67); (1.6, 3.5); (1.8, 0.94); (2.0, 2.0)]
interpolation = [Newtonian_Polynomial(data, 0.01 * i) for i = 0 : 200]
# println(interpolation)
plot(interpolation)
