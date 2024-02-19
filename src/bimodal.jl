module bimodal

export path

function path(T,dt,par,decpar,rates)
	# number of time-steps
	N = ceil(Int, T / dt)

	#parameters
	a = par[1]
	b = par[2]
	c = par[3]
	w = par[4]

	# learning rates
	eta_a = rates[1]
	eta_b = rates[2]
	eta_c = rates[3]
	eta_w = rates[4]

	# noise sources
	dW = sqrt(dt) .* randn(N)
	dV = sqrt(dt) .* randn(N)

	# state and observation
	X = zeros(N)
	dY = zeros(N)

	# filter
	mu = zeros(N)
	P = zeros(N)

	# filter derivatives
	mua = zeros(N)
	mub = zeros(N)
	muc = zeros(N)
	muw = zeros(N)
	Pa = zeros(N)
	Pb = zeros(N)
	Pc = zeros(N)
	Pw = zeros(N)

	# parameter estimates
	af = zeros(N)
	bf = zeros(N)
	cf = zeros(N)
	wf = zeros(N)

	# initialize parameter estimates
	af[1] = decpar[1]
	bf[1] = decpar[2]
	cf[1] = decpar[3]
	wf[1] = decpar[4]

	# special initial conditions
	X[1] = 0
	P[1] = 0
	Pa[1] = 0
	Pb[1] = 0
	Pc[1] = 0
	Pw[1] = 0

	# mean-squared error
	mse = zeros(N)
	mse[1] = X[1]^2

	# Euler-Maruyama integration of stochastic variables
	for t = 2:N
		# state and observation
		X[t] = c*dW[t] + X[t-1] + dt*X[t-1]*(a - b*X[t-1]^2)
		dY[t]= dV[t] + dt*w*X[t-1]

		# filter
		mu[t] = mu[t-1] + dt*af[t-1]*mu[t-1] - dt*bf[t-1]*mu[t-1]^3 - 3*dt*bf[t-1]*mu[t-1]*P[t-1] + dY[t]*P[t-1]*wf[t-1] - dt*mu[t-1]*P[t-1]*wf[t-1]^2
		P[t] = dt*cf[t-1]^2 + P[t-1] + 2*dt*af[t-1]*P[t-1] - 6*dt*bf[t-1]*mu[t-1]^2*P[t-1] - 6*dt*bf[t-1]*P[t-1]^2 - dt*P[t-1]^2*wf[t-1]^2

		# filter derivatives
		mua[t] = -3*dt*bf[t-1]*mu[t-1]^2*mua[t-1] + dY[t]*Pa[t-1]*wf[t-1] - dt*mu[t-1]*(-1 + Pa[t-1]*(3*bf[t-1] + wf[t-1]^2)) + mua[t-1]*(1 + dt*(af[t-1] - P[t-1]*(3*bf[t-1] + wf[t-1]^2)))
		mub[t] = -(dt*mu[t-1]^3) - 3*dt*bf[t-1]*mu[t-1]^2*mub[t-1] + dY[t]*Pb[t-1]*wf[t-1] - dt*mu[t-1]*(3*P[t-1] + Pb[t-1]*(3*bf[t-1] + wf[t-1]^2)) + mub[t-1]*(1 + dt*(af[t-1] - P[t-1]*(3*bf[t-1] + wf[t-1]^2)))
		muc[t] = Pc[t-1]*(dY[t]*wf[t-1] - dt*mu[t-1]*(3*bf[t-1] + wf[t-1]^2)) + muc[t-1]*(1 + dt*(af[t-1] - 3*bf[t-1]*(mu[t-1]^2 + P[t-1]) - P[t-1]*wf[t-1]^2))
		muw[t] = dY[t]*(P[t-1] + Pw[t-1]*wf[t-1]) - dt*mu[t-1]*(2*P[t-1]*wf[t-1] + Pw[t-1]*(3*bf[t-1] + wf[t-1]^2)) + muw[t-1]*(1 + dt*(af[t-1] - 3*bf[t-1]*(mu[t-1]^2 + P[t-1]) - P[t-1]*wf[t-1]^2))
		Pa[t] = (1 + 2*dt*af[t-1] - 6*dt*bf[t-1]*mu[t-1]^2)*Pa[t-1] - 2*dt*P[t-1]*(-1 + 6*bf[t-1]*(mu[t-1]*mua[t-1] + Pa[t-1]) + Pa[t-1]*wf[t-1]^2)
		Pb[t] = Pb[t-1] - 2*dt*(3*P[t-1]*(mu[t-1]^2 + 2*bf[t-1]*mu[t-1]*mub[t-1] + P[t-1]) - af[t-1]*Pb[t-1] + 3*bf[t-1]*(mu[t-1]^2 + 2*P[t-1])*Pb[t-1] + P[t-1]*Pb[t-1]*wf[t-1]^2)
		Pc[t] = 2*dt*cf[t-1] - 6*dt*bf[t-1]*(2*mu[t-1]*muc[t-1]*P[t-1] + (mu[t-1]^2 + 2*P[t-1])*Pc[t-1]) + Pc[t-1]*(1 + 2*dt*af[t-1] - 2*dt*P[t-1]*wf[t-1]^2)
		Pw[t] = -6*dt*bf[t-1]*(2*mu[t-1]*muw[t-1]*P[t-1] + (mu[t-1]^2 + 2*P[t-1])*Pw[t-1]) - 2*dt*P[t-1]^2*wf[t-1] + Pw[t-1]*(1 + 2*dt*af[t-1] - 2*dt*P[t-1]*wf[t-1]^2)

		# parameter estimates
		af[t] = af[t-1]+eta_a*af[t-1]*(mua[t-1]*wf[t-1])*(dY[t]-(mu[t-1]*wf[t-1])dt)
		bf[t] = bf[t-1]+eta_b*bf[t-1]*(mub[t-1]*wf[t-1])*(dY[t]-(mu[t-1]*wf[t-1])dt)
		cf[t] = cf[t-1]+eta_c*cf[t-1]*(muc[t-1]*wf[t-1])*(dY[t]-(mu[t-1]*wf[t-1])dt)
		wf[t] = wf[t-1]+eta_w*wf[t-1]*(mu[t-1] + muw[t-1]*wf[t-1])*(dY[t]-(mu[t-1]*wf[t-1])dt)

		# mean-squared error
		mse[t] = (X[t]-mu[t])^2
	end

	# normalize mse
	mse = mse / 1.1701
	return 0:dt:((N-1)*dt), X, dY, mu, P, af, bf, cf, wf, mse
end

end #module kalmanbucy
