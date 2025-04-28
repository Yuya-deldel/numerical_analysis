# numerical integration 
# trapezoidal rule
# I = h/2 * (f0 + fn + 2(f1 + ... + fn-1))
function integration_trapezoid(func, interval, n)
    (a, b) = interval
    h = (b - a) / n 
    T = (func(a) + func(b)) / 2.0
    for i = 1 : n - 1
        T += func(a + i * h)
    end
    return T * h 
end

function integration_Romberg(func, interval, ϵ, N_max=10)
    (a, b) = interval
    h = b - a
    T = zeros(N_max + 1)
    T[1] = h * (func(a) + func(b)) / 2.0
    
    for n = 1 : N_max
        h /= 2.0
        tmp = (func(a) + func(b)) / 2.0
        for j = 1 : 2^n - 1
            tmp += func(a + j * h)
        end
        T[n+1] = tmp * h 

        if abs(T[n+1] - T[n]) < ϵ 
            return T[n+1]
        end

        # extrapolation
        k = n + 1
        for m = 1 : n 
            k -= 1
            T[k] = (4.0^m * T[k+1] - T[k]) / (4.0^m - 1)
            if k > 1 && abs(T[k] - T[k-1]) < ϵ
                return T[k]
            end
        end
    end
    return T[N_max + 1]
end

# test 
f(x) = 2.0 / x^2 
println(integration_Romberg(f, (1, 2), 0.00000001))

g(x) = 4.0 / (1.0 + x^2)
println(integration_Romberg(g, (0, 1), 0.00000001))