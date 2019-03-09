#=
INPUT : n - a vector containing the number of points on each dimention --> output grid
		h - a vector containing the spaceing of points on each dimention --> output grid
		x - a "matrix" of size  containing the values on the points --> input grid
		y - a place holder of size n1*n2 for the answer
		op - the operator of intrest (interolation)

=#

function interolation_Dirichlet!(n,h,x::Array,y::Array)
	n1_fine = n[1]-1
	n2_fine = n[2]-1
	n1_coarse = Int((n1_fine/2))
	n2_coarse = Int((n2_fine/2))
	println(n1_coarse)
	println(n1_fine)

# coppy common points
	for i=1:n2_coarse
		colShift_fine = (i-1)*n1_fine
		colShift_coarse = (i-1)*n1_coarse
		for j=1:n1_coarse
		print("coarse grid [$(colShift_coarse + j)] --> fine grid [$(2*colShift_fine + 2*j - 1)] \n")
		y[2*colShift_fine + 2*j - 1] = x[colShift_coarse + j]

		end
	end

end

n_fine = [2^3+1,2^3+1];
h_fine = 1.0./n;
n_coarse = (n_fine.-1)/2
n_coarse = [Int(x)+1 for x in n_coarse]
h_coarse = h_fine*2

x_fine = randn(tuple((n_fine.-1)...))
x_coarse = randn(tuple((n_coarse.-1)...))

interolation_Dirichlet!(n_fine,h_fine,x_coarse,x_fine)
