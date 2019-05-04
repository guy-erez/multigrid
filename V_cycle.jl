using LinearAlgebra
using Plots
pyplot()

function MG_V_cycle!(n_cells::Array, h::Array, initial_guss::Array, w, f::Array, output::Array, iter_num::Int)
        n_cells_coarse = [Int(n_cells[1]/2),Int(n_cells[2]/2)]
        # place holders:
        fine_error =  randn(tuple((n_cells.-1)...))
        output_laplasian = randn(tuple((n_cells.-1)...))
        coarse_error = randn(tuple((n_cells_coarse.-1)...))
        residual = randn(tuple((n_cells_coarse.-1)...))

        jacobi!(n_cells,h,initial_guss,w,f,output,iter_num)
        initial_guess = output

        if(n_cells[1]%2 == 0 && n_cells[2]%2 == 0 && n_cells[1] > 4 && n_cells[2] > 4 ) # there are more levels to go
                #compute the residual
                multOpDirichlet!(n_cells,h,initial_guess,output_laplasian,Apply2DLaplasian,w,f) # f is not in use in Apply2DLaplasian #
                fine_residual = f - output_laplasian
                full_weighting_Dirichlet!(n_cells_coarse,h*2,fine_residual,residual)
                MG_V_cycle!(n_cells_coarse, h*2, fill(0.0,tuple((n_cells.-1)...)), w, residual, coarse_error, iter_num)
                interpolation_Dirichlet!(n_cells, h, coarse_error, fine_error)
                initial_guess = initial_guess + fine_error
        end
        jacobi!(n_cells,h,initial_guss,w,f,output,iter_num)

end

function test_MG_V_cycle()
    n = [2^8,2^8];
    h = 1.0./n;
    initial_guss = randn(tuple((n.-1)...))
    output = randn(tuple((n.-1)...))
    output_laplasian = randn(tuple((n.-1)...))
    f = fill(0.0,tuple((n.-1)...))
    w = 4.0/5.0

    MG_V_cycle!(n,h,initial_guss,w,f,output,100)
    multOpDirichlet!(n,h,output,output_laplasian,Apply2DLaplasian,w,f)   # laplasian on the output of MG_2 into "output_laplasian"
    residual = output_laplasian - f
    println("done MG_V_cycle")
     #display(output_jacobi)
    println("the abs of the residual norm is $(norm(residual))")

    #ploting

    x = 1:n[1]-1
    y= 1:n[2]-1

    plot(x,y,output,st=:surface,camera=(-30,30))
    savefig("plot_fig_mg_v_cycle")
end

println("start MG_V_cycle")
test_MG_V_cycle()
