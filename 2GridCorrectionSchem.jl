
#=
n_cells = [n1,n2] , the dim of the grid
h = [h1,h2] , the spaceing of the grid
output - a place holder for the answer size of n1-1*n2-1
initial_guss - size of n1-1*n2-1
=#
function MG_2(n_cells::Array, h::array, output::Array, initial_guss::Array, f::Array, iter_num::int)
    output_jacobi = randn(tuple((n_cells.-1)...))
    output_laplasian = randn(tuple((n_cells.-1)...))

    jacobi!(n_cells,h,initial_guss,output_jacobi,iter_num,multOpDirichlet!)      # relax on fine grid
    multOpDirichlet!(n_cells,h,output_jacobi,output_laplasian,Apply2DLaplasian)  # compute outp <- laplas(output_jacobi) with Diriclet boundry conditions

    fine_residual = f - output_laplasian
    full_weighting_Dirichlet!(n_cells/2,h*2,fine_residual,coarse_residual)
end
