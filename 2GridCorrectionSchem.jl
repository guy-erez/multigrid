
"""
    MG_2!(n_cells,h,initial_guess,w,f,output,iter_num)

sole Lu=f (laplace equation) useing a two-grid correction scheme
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
function MG_2!(n_cells::Array, h::Array, initial_guess::Array, w, f::Array, output::Array, iter_num::Int)
    n_cells_coarse = [Int(n_cells[1]/2),Int(n_cells[2]/2)]
    output_laplasian = randn(tuple((n_cells.-1)...))
    coarse_error =  randn(tuple((n_cells_coarse.-1)...))
    fine_error = randn(tuple((n_cells.-1)...))
    coarse_residual = randn(tuple((n_cells_coarse.-1)...))

     # relax on fine grid
    jacobi!(n_cells,h,initial_guess,w,f,output,1)

    # compute the fine residual
    # compute: output_laplasian <- Laplas(output) with Diriclet boundry conditions
    multOpDirichlet!(n_cells,h,output,output_laplasian,Apply2DLaplacian,w,f) # f is not in use in Apply2DLaplacian #
    fine_residual = f - output_laplasian

    # restrict the residual to the coarse grid
    full_weighting_Dirichlet!(n_cells_coarse,h*2,fine_residual,coarse_residual)

    # solve: laplasin(error) = residual
    jacobi!(n_cells_coarse,h*2,zeros((n_cells_coarse.-1)...),w,coarse_residual,coarse_error,iter_num)

    # Interpolate the coarse-grid error to the fine grid
    interpolation_Dirichlet!(n_cells,h,coarse_error,fine_error)

    initial_guess = output + fine_error
    jacobi!(n_cells,h,initial_guess,w,f,output,1)
end
