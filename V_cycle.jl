
"""
    MG_V_cycle!(n_cells,h,initial_guess,w,f,output,iter_num)

sole Lu=f (laplace equation) useing a V cycle correction scheme
...
# Arguments
- `n_cells::Array`:[n1,n2] the number of cells in the wanted grid
- `h::Array`:[h1,h2] the spaceing length in the grid
- `initial_guess::Array`: (n1-1xn2-1) initial values
- `w::Float64`: weight (usually 4/5)
- `f::Array`:(n1-1xn2-1) values of f
- `output::Array`: (n1-1xn2-1) place holder
- `iter_num::Int`: the number of Jacobi iterations to be done on the bottom level
...

"""
function MG_V_cycle!(n_cells::Array, h::Array, initial_guess::Array, w, f::Array, output::Array, iter_num::Int)
        n_cells_coarse = [Int(n_cells[1]/2),Int(n_cells[2]/2)]
        # place holders:
        fine_error =  randn(tuple((n_cells.-1)...))
        output_laplasian = randn(tuple((n_cells.-1)...))
        coarse_error = randn(tuple((n_cells_coarse.-1)...))
        residual = randn(tuple((n_cells_coarse.-1)...))

        jacobi!(n_cells,h,initial_guess,w,f,output,2)
        initial_guess = output

        if(n_cells[1]%2 == 0 && n_cells[2]%2 == 0 && n_cells[1] > 4 && n_cells[2] > 4 ) # there are more levels to go
                #compute the residual
                multOpDirichlet!(n_cells,h,initial_guess,output_laplasian,Apply2DLaplasian,w,f) # f is not in use in Apply2DLaplasian #
                fine_residual = f - output_laplasian
                full_weighting_Dirichlet!(n_cells_coarse,h*2,fine_residual,residual)

                MG_V_cycle!(n_cells_coarse, h*2, fill(0.0,tuple((n_cells.-1)...)), w, residual, coarse_error, iter_num)

                interpolation_Dirichlet!(n_cells, h, coarse_error, fine_error)
                initial_guess = initial_guess + fine_error
        else
                jacobi!(n_cells,h,initial_guess,w,f,output,iter_num)
        end
        jacobi!(n_cells,h,initial_guess,w,f,output,2)
end
