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
####### from matrix notation to julia : x[i,j] = x[i + n1*(j-1)]

# coppy common points y[2i-1,2j-1]=x[i,j]
	for j=1:n2_coarse
		for i=1:n1_coarse
			y[2*i - 1 + (2*j-2)*n1_fine] = x[i + (j-1)*n1_coarse]
		end
	end

#y[2i,2j-1] <- (x[i,j]+x[i+1,j])/2
	for j=1:n2_coarse
		for i=1:n1_coarse-1
			y[2*i + (2*j-2)*n1_fine]=(x[i + (j-1)*n1_coarse] + x[i + 1 + (j-1)*n1_coarse])/2
		end
		y[n1_fine + (2*j-2)*n1_fine]=(x[n1_coarse + (j-1)*n1_coarse] + 0)/2 			#last point on each column
	end

#y[2i-1,2j] <- (x[i,j]+x[i,j+1])/2
	for j=1:n2_coarse-1
		for i=1:n1_coarse
			y[(2*i - 1) + (2*j-1)*n1_fine]=(x[i + (j-1)*n1_coarse] + x[i + j*n1_coarse])/2
		end
	end
# y[2i-1,n2_fine] <- (x[i,n2_coarse]+0)/2 last point on each row
	for i=1:n1_coarse
		y[(2*i - 1) + (n2_fine-1)*n1_fine] = (x[i + (n2_coarse-1)*n1_coarse] + 0)/2
	end

#y[2i,2j] <- (x[i,j] + x[i+1,j] + x[i,j+1] + x[i+1,j+1])/4
	for j=1:n2_coarse-1
		for i=1:n1_coarse-1
			y[2*i + (2*j-1)*n1_fine] = (x[i + (j-1)*n1_coarse] + x[(i+1) + (j-1)*n1_coarse] + x[i + j*n1_coarse] + x[(i+1) + j*n1_coarse])/4
		end
		# last point on the column
		i = n1_coarse
		y[2*i + (2*j-1)*n1_fine] = (x[i + (j-1)*n1_coarse] + 0 + x[i + j*n1_coarse] + 0)/4
	end

#y[2i,2j] <- (x[i,j] + x[i+1,j] + x[i,j+1] + x[i+1,j+1])/4 last point on each row
	for i=1:n1_coarse-1
		y[2*i + (n2_fine-1)*n1_fine] = (x[i + (n2_coarse-1)*n1_coarse] + x[(i+1) + (n2_coarse-1)*n1_coarse] + 0 + 0)/4
	end
#last point y[n1_fine,n2_fine] = x[n1_coarse,n2_coarse]/4
	y[n1_fine + (n2_fine-1)*n1_fine] = x[n1_coarse + (n2_coarse-1)*n1_coarse]/4
end

n_fine = [2^3+1,2^3+1];
h_fine = 1.0./n_fine;
n_coarse = (n_fine.-1)/2
n_coarse = [Int(x)+1 for x in n_coarse]
h_coarse = h_fine*2

x_fine = randn(tuple((n_fine.-1)...))
x_coarse = randn(tuple((n_coarse.-1)...))

interolation_Dirichlet!(n_fine,h_fine,x_coarse,x_fine)
println("done")
