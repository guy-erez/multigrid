using Plots
pyplot()

function inner_loop(u11,u21,u12,u22)
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

#=
INPUT : n - a vector containing the number of points on each dimention --> output grid
		h - a vector containing the spaceing of points on each dimention --> output grid
		x - a "matrix" of size  containing the values on the points --> input grid
		y - a place holder of size n1*n2 for the answer
		op - the operator of intrest (interolation)

=#

function interolation_Dirichlet!(n_cells,h,x::Array,y::Array)
	n1_fine = n_cells[1]
	n2_fine = n_cells[2]
	n1_coarse = Int((n1_fine/2))-1
	n2_coarse = Int((n2_fine/2))-1
	n1_fine = Int(n1_fine-1)
	n2_fine = Int(n2_fine-1)
#each
####### from matrix notation to julia : x[i,j] = x[i + n1*(j-1)]##########
#first coluumn
	(~,~,~,y[1]) = inner_loop(0.0,0.0,0.0,x[1])
	for i = 1 : n1_coarse-1
		(~,~,y[2*i],y[2*i + 1]) = inner_loop(0,0,x[i],x[i + 1])
	end
	(~,~,y[n1_fine-1],y[n1_fine]) = inner_loop(0.0,0.0,x[n1_coarse],0.0)

	for j = 1 : n2_coarse - 1
		(~,y[1 + (2*j - 1)*n1_fine],~,y[1 + (2*j)*n1_fine]) = inner_loop(0.0,x[1 + (j-1)*n1_coarse],0.0,x[1 + (j)*n1_coarse])
		for i = 1 : n1_coarse - 1
			(y[2*i + (2*j - 1)*n1_fine],y[(2*i + 1) + (2*j - 1)*n1_fine],y[2*i + (2*j)*n1_fine],y[(2*i + 1)+ (2*j)*n1_fine]) =
			inner_loop(x[i + (j-1)*n1_coarse],x[(i + 1) + (j-1)*n1_coarse],x[i + (j)*n1_coarse],x[(i + 1) + (j)*n1_coarse])
		end
		i = n1_coarse
		(y[2*i + (2*j - 1)*n1_fine],y[(2*i + 1) + (2*j - 1)*n1_fine],y[2*i + (2*j)*n1_fine],y[(2*i + 1)+ (2*j)*n1_fine]) =
		inner_loop(x[i + (j-1)*n1_coarse],0.0,x[i + (j)*n1_coarse],0.0)
	end

#last column
	(~,y[1 + (n2_fine - 2)*n1_fine],~,y[1 + (n2_fine - 1)*n1_fine]) = inner_loop(0.0,x[1 + (n2_coarse-1)*n1_coarse],0.0,0.0)
	for i = 1 : n1_coarse-1
		(y[2*i + (n2_fine - 2)*n1_fine] , y[(2*i + 1) + (n2_fine - 2)*n1_fine] , y[2*i + (n2_fine - 1)*n1_fine] , y[(2*i + 1) + (n2_fine - 1)*n1_fine]) =
		inner_loop(x[i + (n2_coarse - 1)*n1_coarse],x[(i + 1) + (n2_coarse - 1)*n1_coarse], 0.0 , 0.0)
	end
	(y[(n1_fine - 1) + (n2_fine -2 )*n1_fine] , y[n1_fine + (n2_fine -2 )*n1_fine] , y[(n1_fine - 1) + (n2_fine - 1 )*n1_fine] , y[n1_fine*n2_fine] ) =
	inner_loop(x[n1_coarse*n2_coarse],0.0,0.0,0.0)
end

n_fine = [2^4,2^5];
h_fine = 1.0./n_fine;
n_coarse = (n_fine)/2
n_coarse = [Int(x)+1 for x in n_coarse]
h_coarse = h_fine*2

x_fine = ones(tuple((n_fine.-1)...))
x_fine = x_fine .* 5
x_coarse = ones(tuple((n_coarse.-1)...))

interolation_Dirichlet!(n_fine,h_fine,x_coarse,x_fine)
println("done")

y1 = 1:n_fine[1]-1
x1 = 1:n_fine[2]-1
p1 = plot(x1,y1,x_fine,st=:surface,camera=(-30,30))

y2 = 1:n_coarse[1]-1
x2 = 1:n_coarse[2]-1
p2 = plot(x2,y2,x_coarse,st=:surface,camera=(-30,30))

plot(p1,p2,layout=(1,2),legend=false)
savefig("plot_fig")


p3 = plot(y1,x_fine[:,1])
p4 = plot(y1,x_fine[:,(n_fine[2]-1)])
p5 = plot(x1,x_fine[1,:])
p6 = plot(y1,x_fine[(n_fine[1]-1),:])   ###something went wrong, but the values seems fine

plot(p3,p4,p5,p6,layout=(2,2),legend=false)
savefig("edges")
