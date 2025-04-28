# Ordinary Differential Equations solver
# Heun method 
function Heun_method(func, initial_value, interval, n)
    (a, b) = interval
    h = (b - a) / n 
    Y = zeros(n + 1)
    Y[1] = initial_value 
    x = a
    for i = 1 : n 
        k_1 = func(x, Y[i])
        k_2 = func(x + h, Y[i] + h * k_1)
        Y[i+1] = Y[i] + h * (k_1 + k_2) / 2.0
        x += h 
    end
    return Y
end

# Runge Kutta method (4th order)
function Runge_Kutta_4(func, initial_value, interval, n)
    (a, b) = interval
    h = (b - a) / n 
    Y = zeros(n + 1)
    Y[1] = initial_value 
    x = a
    for i = 1 : n 
        k_1 = func(x, Y[i])
        k_2 = func(x + h / 2.0, Y[i] + h / 2.0 * k_1)
        k_3 = func(x + h / 2.0, Y[i] + h / 2.0 * k_2)
        k_4 = func(x + h, Y[i] + h * k_3)
        Y[i+1] = Y[i] + h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0
        x += h 
    end
    return Y
end

# Predictor_Corrector_Method: small partition number(n) -> iteratively improving solution
# Adams method (4th order)
function Adams_method(func, initial_value, interval, ϵ, n, Max=10)
    (a, b) = interval 
    h = (b - a) / n 
    
    # starter: by Runge_Kutta_method
    Y = Runge_Kutta_4(func, initial_value, interval, n)
    F = [func(a, Y[1]); func(a + h, Y[2]); func(a + 2.0 * h, Y[3]); 0.0; 0.0] 
    x = a + 4.0 * h 

    for i = 4 : n 
        # Adams_Bashforth method 
        F[4] = func(x-h, Y[i])
        y_pred = Y[i] + h * (55.0 * F[4] - 59.0 * F[3] + 37.0 * F[2] - 9.0 * F[1]) / 24.0 
        for j = 1 : Max 
            # Adams_Moulton method 
            F[5] = func(x, y_pred)
            Y[i+1] = Y[i] + h * (9.0 * F[5] + 19.0 * F[4] - 5.0 * F[3] + F[2]) / 24.0 

            if abs(Y[i+1] - y_pred) < ϵ 
                break 
            end
            y_pred = Y[i+1]
        end
        for j = 1 : 4 
            F[j] = F[j+1]
        end
        x += h 
    end
    return Y 
end

# test 
f(x, y) = x + y 
println(Adams_method(f, 1.0, (0.0, 1.0), 0.00000001, 10))