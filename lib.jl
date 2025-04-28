# 1_norm for vectors
function norm(vec)
    n = size(vec, 1)
    sum = 0
    for i = 1 : n 
        sum += abs(vec[i])
    end
    return sum     
end

# 2_norm for vectors 
function norm_2(vec)
    n = size(vec, 1)
    sum = 0
    for i = 1 : n 
        sum += vec[i] * vec[i]
    end
    return √sum 
end