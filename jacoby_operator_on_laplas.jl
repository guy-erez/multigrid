#=
PURPOSE : preform an approximation to the laplasian of u at possition u0.
INPUT : u0 - the possition where the laplasian is approximated.
		u_mi = u0 - hi
		u_pi = u0 + hi
		where i denote the directions 1,2
		hi - the grid spaceing in the i directions
OUTPUT : the value of the approximation to the laplasian at u0.
=#

function Apply2DLaplasian(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64) # x-->u
	h1invsq = 1.0/(h1^2)
	h2invsq = 1.0/(h2^2)
	return (2*h1invsq + 2*h2invsq)*u0 - h1invsq*(u_m1 + u_p1) - h2invsq*(u_m2 + u_p2); # wrong sign? d^2f/du^2 = (f(x0-h) - 2f(x0) + f(x0+h)) / h^2
end


function Apply2DJacobi(u0::Float64,u_m1::Float64,u_p1::Float64,u_m2::Float64,u_p2::Float64,h1::Float64,h2::Float64,w::Float64,f::Float64)
	d = -2*(1.0/(h1^2) + 1.0/(h2^2)) 								# the diagonal element of the laplasian operator
	L_u = Apply2DLaplasian(u0,u_m1,u_p1,u_m2,u_p2,h1,h2)			# laplasian*u
	return u0 + w*(1.0/d)*(f - L_u)
end


function multOpDirichlet!(n,h,x::Array,y::Array,w::Float64,f::Array,op::Function)
dim = 2;
if dim==2
	n1 = n[1]-1;
	n2 = n[2]-1;
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




n = [10,10];
h = 1.0./n;
u = randn(tuple((n.-1)...))
y = randn(tuple((n.-1)...))
f = randn(tuple((n.-1)...))
w = 2.0/3.0

multOpDirichlet!(n,h,u,y,w,f,Apply2DJacobi);

println(y)
