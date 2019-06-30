using LinearAlgebra
using Plots
pyplot()
#=
n_cells = [n1,n2] , the dim of the grid
h = [h1,h2] , the spaceing of the grid
initial_guss - size of n1-1*n2-1
w - weighting for jacobi
f - we solve laplasin(x) = f
output - a place holder for the answer size of n1-1*n2-1
iter_num - number of iterations on each grid
=#
function MG_2!(n_cells::Array, h::Array, initial_guss::Array, w, f::Array, output::Array, iter_num::Int)
    n_cells_coarse = [Int(n_cells[1]/2),Int(n_cells[2]/2)]
    output_laplasian = randn(tuple((n_cells.-1)...))
    coarse_error =  randn(tuple((n_cells_coarse.-1)...))
    fine_error = randn(tuple((n_cells.-1)...))
    coarse_residual = randn(tuple((n_cells_coarse.-1)...))

     # relax on fine grid
    jacobi!(n_cells,h,initial_guss,w,f,output,2)

    # compute the fine residual
    # compute: output_laplasian <- Laplas(output) with Diriclet boundry conditions
    multOpDirichlet!(n_cells,h,output,output_laplasian,Apply2DLaplasian,w,f) # f is not in use in Apply2DLaplasian #
    fine_residual = f - output_laplasian

    # restrict the residual to the coarse grid
    full_weighting_Dirichlet!(n_cells_coarse,h*2,fine_residual,coarse_residual)

    # solve: laplasin(error) = residual
    jacobi!(n_cells_coarse,h*2,zeros((n_cells_coarse.-1)...),w,coarse_residual,coarse_error,iter_num)

    # Interpolate the coarse-grid error to the fine grid
    interpolation_Dirichlet!(n_cells,h,coarse_error,fine_error)

    initial_guss = output + fine_error
    jacobi!(n_cells,h,initial_guss,w,f,output,2)
end


function test_MG_2()
    n = [2^8,2^8];
    h = 1.0./n;
    initial_guss = randn(tuple((n.-1)...))

    output = randn(tuple((n.-1)...))
    output_laplasian = randn(tuple((n.-1)...))
    f = fill(0.0,tuple((n.-1)...))
    w = 4.0/5.0

    println("start MG_2")
    multOpDirichlet!(n,h,initial_guss,output_laplasian,Apply2DLaplasian,w,f)
    residual = output_laplasian - f
    println("init residual  : ||laplasian(output)|| = $(norm(residual))")
    println("init output    : ||output|| =            $(norm(initial_guss))")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for i = 0:20
        MG_2!(n,h,initial_guss,w,f,output,50)
        multOpDirichlet!(n,h,output,output_laplasian,Apply2DLaplasian,w,f)   # laplasian on the output of MG_2 into "output_laplasian"
        residual = output_laplasian - f
        println("~~residual  : ||laplasian(output)|| = $(norm(residual))")
        println("~~output    : ||output|| =            $(norm(output))")
        println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        initial_guss.= output
    end

    println("done MG_2")

    #ploting

    x = 1:n[1]-1
    y= 1:n[2]-1

    plot(x,y,output,st=:surface,camera=(-30,30))
    savefig("plot_fig_mg2")
end

test_MG_2()
