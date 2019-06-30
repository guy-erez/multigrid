using LinearAlgebra
using Plots
pyplot()
"""
    Apply2DLaplacian(u0,u_m1,u_p1,u_m2,u_p2,h1,h2,w,f)

compute an approximation to (-1*)Laplacian of u at possition u0
...
# Arguments
- `u0::Float64`: the value at the wanted position to compute the Laplacian
- `u_m1::Float64`: the value at one cell 'left'
- `u_p1::Float64`: the value at one cell 'right'
- `u_m2::Float64`: the value at one cell 'down'
- `u_p2::Float64`: the value at one cell 'up'
- `h1::Float64`: the spaceing length in first direction
- `h2::Float64`: the spaceing length in second direction
- `w::Float64`: not in use - for generalization
- `f::Float64`: not in use - for generalization
...

"""
function Apply2DLaplacian(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64,w::Float64,f::Float64) # x-->u
	h1invsq = 1.0/(h1^2) # calculted every time with no need
	h2invsq = 1.0/(h2^2) # calculted every time with no need
	return (2*h1invsq + 2*h2invsq)*u0 - h1invsq*(u_m1 + u_p1) - h2invsq*(u_m2 + u_p2);
end

"""
    Apply2DJacobiStep(u0,u_m1,u_p1,u_m2,u_p2,h1,h2,w,f)

compute the Jacobi operation of Lu = f at possition u0 -> u0 + w*(1/2*(1/(h1^2)+1/(h2^2))*(f-Lu)
...
# Arguments
- `u0::Float64`: the value at the wanted position to compute the Laplacian
- `u_m1::Float64`: the value at one cell 'left'
- `u_p1::Float64`: the value at one cell 'right'
- `u_m2::Float64`: the value at one cell 'down'
- `u_p2::Float64`: the value at one cell 'up'
- `h1::Float64`: the spaceing length in first direction
- `h2::Float64`: the spaceing length in second direction
- `w::Float64`: weight (usually 4/5)
- `f::Float64`: f from the equation to be solved Lu=f
...

"""
function Apply2DJacobiStep(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64,w::Float64,f::Float64)
	d = 2*(1.0/(h1^2) + 1.0/(h2^2)) 						  			# the diagonal element of the Laplacian operator
	L_u = Apply2DLaplacian(u0,u_m1,u_p1,u_m2,u_p2,h1,h2,w,f)			# Laplacian*u
	return u0 + w*(1.0/d)*(f - L_u)
end

"""
    jacobi!(n,h,initial_guess,w,f,output,max_iter)

compute max_iter iterations of Jacobi to solve of Lu = f
...
# Arguments
- `n::Array`: [n1,n2] number of cells in the grid
- `h::Float64`: the spaceing length
- `initial_guess::Array`: (n1-1)x(n2-1) initial values
- `u_m2::Float64`: the value at one cell 'down'
- `u_p2::Float64`: the value at one cell 'up'
- `w::Float64`: weight (usually 4/5)
- `f::Float64`: f from the equation to be solved Lu=f
- `output::Array`: (n1-1)x(n2-1) place holder for the result
- `max_iter::Int64`: number of Jacobi iterations to be done
...

"""
function jacobi!(n, h, initial_guess::Array,w::Float64,f::Array, output::Array, max_iter::Int64)
	for i in 1:max_iter
		multOpDirichlet!(n,h,initial_guess,output,Apply2DJacobiStep,w,f)
		initial_guess = output
	end
end

"""
    multOpDirichlet!(n_cells,h,x,y,op,w,f)

compute op operator with zero boundary conditions on x values and save it at y
...
# Arguments
- `n_cells::Array`: [n1,n2] number of cells in the grid
- `h::Float64`: the spaceing length
- `x::Array`: (n1-1)x(n2-1) values as input of the operator
- `y::Array`: (n1-1)x(n2-1) place holder for the result
- `op::Function`: an operator (op(x0,x1,x2,x3,x4,h,w,f0))
- `w::Float64`: weight for op
- `f::Float64`:(n1-1)x(n2-1) values of f for op
...

"""
function multOpDirichlet!(n_cells,h,x::Array,y::Array,op::Function,w,f::Array)
	dim = 2;
	if dim==2
		n1 = n_cells[1]-1;
		n2 = n_cells[2]-1;
		h1 = h[1]
		h2 = h[2]
	# first dot on first column --> zeros to the left and above
		y[1] = op(x[1],0.0,x[2],0.0,x[n1+1],h1,h2,w,f[1]);
		for i=2:n1-1
	# first column --> zeros to the left
			y[i] = op(x[i],x[i-1],x[i+1],0.0,x[i+n1],h1,h2,w,f[i]);
		end
	# last dot on first column --> zeros to the left and below
		y[n1] = op(x[n1],x[n1-1],0.0,0.0,x[n1 + n1],h1,h2,w,f[n1]);
		for j=2:n2-1
			colShift = n1*(j-1);
			i = 1 + colShift;
	# first dot on a column --> zero above
			y[i] = op(x[i],0.0,x[i+1],x[i-n1],x[i+n1],h1,h2,w,f[i]);
			for i = (2 + colShift):(n1-1 + colShift)
				y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],x[i+n1],h1,h2,w,f[i]);
			end
			i = n1 + colShift;
	# last dot on a column --> zero below
			y[i] = op(x[i],x[i-1],0.0,x[i-n1],x[i+n1],h1,h2,w,f[i]);
		end
		colShift = n1*(n2-1);
		i = 1 + colShift;
	# first dot on last column --> zeros to the right and above
		y[i] = op(x[i],0.0,x[i+1],x[i-n1],0.0,h1,h2,w,f[i]);
		for i = (2 + colShift):(n1-1 + colShift)
	# last column --> zeros to the right
			y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],0.0,h1,h2,w,f[i]);
		end
		i = n1 + colShift;
	# last dot on last column --> zero to the right and below
		y[i] = op(x[i],x[i-1],0.0,x[i-n1],0.0,h1,h2,w,f[i]);
	else
	end
end
