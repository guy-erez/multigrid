####### from matrix notation to julia : x[i,j] = x[i + n1*(j-1)]##########

function inner_loop_interpolation(u11,u21,u12,u22)
	#y[2i,2j]=x[i,j] coppy common points
	y11 = u11
	#y[2i+1,2j] <- (x[i,j]+x[i+1,j])/2
	y21=(u11 + u21)/2.0
	#y[2i,2j+1] <- (x[i,j]+x[i,j+1])/2
	y12 =(u11 + u12)/2.0
	#y[2i+1,2j+1] <- (x[i,j] + x[i+1,j] + x[i,j+1] + x[i+1,j+1])/4
	y22 = (u11 + u12 + u21 +u22)/4
	return (y11,y21,y12,y22)
end

"""
    interpolation_Dirichlet!(n_cells,h,x,y)

interpolat x from coarse grid (n_cells/2) to a fine grid (n_cells)
...
# Arguments
- `n_cells::Array`:[n1,n2] the number of cells in the fine grid
- `h::Array`:[h1,h2] the spaceing length in the fine grid
- `x::Array`: (n1/2xn2/2) values of the function to be interpolated
- `y::Array`: (n1xn2) place holder
...

"""
function interpolation_Dirichlet!(n_cells,h,x::Array,y::Array)
	n1_fine = n_cells[1]
	n2_fine = n_cells[2]
	n1_coarse = Int((n1_fine/2))-1
	n2_coarse = Int((n2_fine/2))-1
	n1_fine = Int(n1_fine-1)
	n2_fine = Int(n2_fine-1)

	# first column - first node #
	(~,~,~,y[1]) =
	inner_loop_interpolation(0.0,0.0,0.0,x[1])
	# last column - first node #
	(~,y[1 + (n2_fine - 2)*n1_fine],~,y[1 + (n2_fine - 1)*n1_fine]) =
	inner_loop_interpolation(0.0,x[1 + (n2_coarse-1)*n1_coarse],0.0,0.0)
	# first coluumn - last node #
	(~,~,y[n1_fine-1],y[n1_fine]) =
	inner_loop_interpolation(0.0,0.0,x[n1_coarse],0.0)
	# last column - last node #
	(y[(n1_fine - 1) + (n2_fine - 2)*n1_fine] , y[n1_fine + (n2_fine -2 )*n1_fine] , y[(n1_fine - 1) + (n2_fine - 1 )*n1_fine] , y[n1_fine*n2_fine] ) =
	inner_loop_interpolation(x[n1_coarse*n2_coarse],0.0,0.0,0.0)

	for i = 1 : n1_coarse-1
		# first coluumn #
		(~,~,y[2*i],y[2*i + 1]) =
		inner_loop_interpolation(0,0,x[i],x[i + 1])
		# last column #
		(y[2*i + (n2_fine - 2)*n1_fine] , y[(2*i + 1) + (n2_fine - 2)*n1_fine] , y[2*i + (n2_fine - 1)*n1_fine] , y[(2*i + 1) + (n2_fine - 1)*n1_fine]) =
		inner_loop_interpolation(x[i + (n2_coarse - 1)*n1_coarse],x[(i + 1) + (n2_coarse - 1)*n1_coarse], 0.0 , 0.0) #last column
	end

	# inner nodes #

	for j = 1 : n2_coarse - 1
		# first node on each column #
		(~,y[1 + (2*j - 1)*n1_fine],~,y[1 + (2*j)*n1_fine]) =
		inner_loop_interpolation(0.0,x[1 + (j-1)*n1_coarse],0.0,x[1 + (j)*n1_coarse])
		for i = 1 : n1_coarse - 1
			(y[2*i + (2*j - 1)*n1_fine],y[(2*i + 1) + (2*j - 1)*n1_fine],y[2*i + (2*j)*n1_fine],y[(2*i + 1)+ (2*j)*n1_fine]) =
			inner_loop_interpolation(x[i + (j-1)*n1_coarse],x[(i + 1) + (j-1)*n1_coarse],x[i + (j)*n1_coarse],x[(i + 1) + (j)*n1_coarse])
		end
		# last nose on each column #
		i = n1_coarse
		(y[2*i + (2*j - 1)*n1_fine],y[(2*i + 1) + (2*j - 1)*n1_fine],y[2*i + (2*j)*n1_fine],y[(2*i + 1)+ (2*j)*n1_fine]) =
		inner_loop_interpolation(x[i + (j-1)*n1_coarse],0.0,x[i + (j)*n1_coarse],0.0)
	end

end

#=
INPUT : n - a vector containing the number of cells on each dimention --> output grid
		h - a vector containing the spaceing of nodes on each dimention --> output grid
		x - a "matrix" of size (n1/2)-1*(n2/2)-1 containing the values on the points --> fine grid
		y - a place holder of size n1-1*n2-1 for the answer  --> coarse grid
=#
"""
    full_weighting_Dirichlet!(n_cells,h,x,y)

restrict x from fine grid (n_cells*2) to a coarse grid (n_cells)
...
# Arguments
- `n_cells::Array`:[n1,n2] the number of cells in the coarse grid
- `h::Array`:[h1,h2] the spaceing length in the coarse grid
- `x::Array`: (n1*2xn2*2) values of the function to be restricted
- `y::Array`: (n1xn2) place holder
...

"""
function full_weighting_Dirichlet!(n_cells,h,x::Array,y::Array)
	n1_coarse = n_cells[1] - 1
	n2_coarse = n_cells[2] - 1
	n1_fine = Int((n_cells[1]*2))-1
	n2_fine = Int((n_cells[2]*2))-1
	for j = 1 : n2_coarse
		for i = 1 : n1_coarse
			y[i + (j-1)*n1_coarse] =
			1/16*(x[(2*i - 1) + (2*j - 2)*n1_fine] + x[(2*i - 1) + (2*j)*n1_fine] + x[(2*i + 1) + (2*j - 2)*n1_fine] + x[(2*i + 1) + (2*j)*n1_fine]+
			2*(x[(2*i) + (2*j - 2)*n1_fine] + x[(2*i) + (2*j)*n1_fine] + x[(2*i - 1) + (2*j - 1)*n1_fine] + x[(2*i + 1) + (2*j - 1)*n1_fine])+
			4*(x[2*i + (2*j-1)*n1_fine]))
		end
	end
end
