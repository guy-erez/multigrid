using LinearAlgebra
using Plots
pyplot()
#=
PURPOSE : preform an approximation to the (-)laplasian of u at possition u0.
INPUT : u0 - the possition where the laplasian is approximated.
		u_mi = u0 - hi
		u_pi = u0 + hi
		where i denote the directions 1,2
		hi - the grid spaceing in the i directions
		w,f are not used in this
OUTPUT : the value of the approximation to the laplasian at u0.
=#

function Apply2DLaplasian(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64,w,f) # x-->u
	h1invsq = 1.0/(h1^2) # calculted every time with no need
	h2invsq = 1.0/(h2^2) # calculted every time with no need
	return (2*h1invsq + 2*h2invsq)*u0 - h1invsq*(u_m1 + u_p1) - h2invsq*(u_m2 + u_p2);
end


function Apply2DJacobiStep(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64,w::Float64,f::Float64)
	d = 2*(1.0/(h1^2) + 1.0/(h2^2)) 						  			# the diagonal element of the laplasian operator
	L_u = Apply2DLaplasian(u0,u_m1,u_p1,u_m2,u_p2,h1,h2,w,f)			# laplasian*u
	return u0 + w*(1.0/d)*(f - L_u)
end


function jacobi!(n, h, initial_guss::Array, output::Array, max_iter ::Int  ,boundry_condition :: Function)
	for i in 1:max_iter
		boundry_condition(n,h,initial_guss,output,Apply2DJacobiStep)
		initial_guss = output
	end
end


function multOpDirichlet!(n_cells,h,x::Array,y::Array,op::Function)
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


n = [100,100];
h = 1.0./n;
u = randn(tuple((n.-1)...))
output_jacobi = randn(tuple((n.-1)...))
output_laplasian = randn(tuple((n.-1)...))
f = fill(0.0,tuple((n.-1)...))
w = 4.0/5.0

jacobi!(n,h,u,output_jacobi,1200,multOpDirichlet!)
multOpDirichlet!(n,h,output_jacobi,output_laplasian,Apply2DLaplasian)   # laplasian on the output of jacoi into "output_laplasian"
residual = output_laplasian - f

println("\nthe approximate value is :")
 #display(output_jacobi)
println("\nthe abs of the residual norm is $(norm(residual))\n")

#ploting

x = 1:n[1]-1
y= 1:n[2]-1

plot(x,y,output_jacobi,st=:surface,camera=(-30,30))
savefig("plot_fig")
