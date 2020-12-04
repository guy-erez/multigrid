using LinearAlgebra
using Plots
pyplot()

n = [2^8,2^8];
h = 1.0./n;
output_laplasian = zeros(tuple((n.-1)...))
f = zeros(tuple((n.-1)...))
w = 4.0/5.0

iterations = 21
total_jacoi_iter = 500
initial_guess = randn(tuple((n.-1)...))

# cllasic jacobi
residual_j = zeros(iterations)
output_norm_j = zeros(iterations)
output_jacobi = zeros(tuple((n.-1)...))
initial_guess_j = zeros(tuple((n.-1)...))
initial_guess_j .= initial_guess

for i = 2:iterations
    jacobi!(n,h,initial_guess_j,w,f,output_jacobi,floor(Int,total_jacoi_iter/(iterations-1)))
    multOpDirichlet!(n,h,output_jacobi,output_laplasian,Apply2DLaplacian,w,f)   # laplasian on the output of jacobi into "output_laplasian"
    residual_j[i] = norm(output_laplasian - f)
    output_norm_j[i] = norm(output_jacobi)

    initial_guess_j.= output_jacobi
end

# V cycle
residual_v = zeros(iterations)
output_norm_v = zeros(iterations)
initial_guess_v = zeros(tuple((n.-1)...))
initial_guess_v .= initial_guess
output_v = zeros(tuple((n.-1)...))

for i = 2:iterations
    MG_V_cycle!(n,h,initial_guess_v,w,f,output_v,floor(Int,(total_jacoi_iter/(iterations-1) - 2*8)))
    multOpDirichlet!(n,h,output_v,output_laplasian,Apply2DLaplacian,w,f)   # laplasian on the output ofMG_V_cycle into "output_laplasian"

    residual_v[i] = norm(output_laplasian - f)
    output_norm_v[i] = norm(output_v)

    initial_guess_v.= output_v
end

# 2 Grid scheme
residual_2 = zeros(iterations)
output_norm_2 = zeros(iterations)
initial_guess_2 = zeros(tuple((n.-1)...))
initial_guess_2 .= initial_guess
output_2 = zeros(tuple((n.-1)...))

for i = 2:iterations
    MG_2!(n,h,initial_guess_2,w,f,output_2,floor(Int,(total_jacoi_iter/(iterations-1) - 2)))
    multOpDirichlet!(n,h,output_2,output_laplasian,Apply2DLaplacian,w,f)   # laplasian on the output of MG_2 into "output_laplasian"

    residual_2[i] = norm(output_laplasian - f)
    output_norm_2[i] = norm(output_2)

    initial_guess_2.= output_2
end

multOpDirichlet!(n,h,initial_guess,output_laplasian,Apply2DLaplacian,w,f)
residual_2[1] = residual_v[1] = residual_j[1] = norm(output_laplasian - f)
output_norm_2[1] = output_norm_v[1] = output_norm_j[1] = norm(initial_guess)


x = range(1, length=iterations, total_jacoi_iter)

p = plot(x,output_norm_v,label="V cycle")
plot!(p,x,output_norm_2,label="2 Grid scheme")
plot!(p,x,output_norm_j,label="jacobi")
title!("Comperation Between Errors")
yaxis!("|| Error ||", :log10)
xlabel!("Iterations")
#ylims!(0,20)
#xlims!(1,100)
savefig("Comperation Between_V_Cycle_and_2_Grid_Scheme_results")

p = plot(x,residual_v,label="V cycle")
plot!(p,x,residual_2,label="2 Grid scheme")
plot!(p,x,residual_j,label="jacobi")
title!("Comperation Between the Residuals")
yaxis!("|| Residual ||", :log10)
xlabel!("Iterations")
#ylims!(0,10^4)
#xlims!(1,)
savefig("Comperation Between_V_Cycle_and_2_Grid_Scheme_residual")



function converg_rate(x)
    output = zeros(length(x)-1)
    for i in 1:(length(x)-1)
        output[i] = x[i+1]/x[i]
    end
    return output
end

x = x[1:20]
con_rate_j = converg_rate(output_norm_j)
con_rate_2 = converg_rate(output_norm_2)
con_rate_v = converg_rate(output_norm_v)

p = plot(x,con_rate_v,label="V cycle")
plot!(p,x,con_rate_2,label="2 Grid scheme")
plot!(p,x,con_rate_j,label="Jacobi")
title!("Comparison Between Convergence Rates")
ylabel!("e(n+1)/e(n)")
xlabel!("Iterations")
savefig("Comparison Between_V_Cycle_and_2_Grid_Scheme_rate")
