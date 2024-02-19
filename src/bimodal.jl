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
		X[t] = c*dW[t] + X[-1 + t] + dt*X[-1 + t]*(a - b*X[-1 + t]^2)
		dY[t]= dV[t] + dt*w*X[-1 + t]

		# filter
		mu[t] = mu[-1 + t] + dt*af[-1 + t]*mu[-1 + t] - dt*bf[-1 + t]*mu[-1 + t]^3 - 3*dt*bf[-1 + t]*mu[-1 + t]*P[-1 + t] + dY[t]*P[-1 + t]*wf[-1 + t] - dt*mu[-1 + t]*P[-1 + t]*wf[-1 + t]^2
		P[t] = dt*cf[-1 + t]^2 + P[-1 + t] + 2*dt*af[-1 + t]*P[-1 + t] - 6*dt*bf[-1 + t]*mu[-1 + t]^2*P[-1 + t] - 6*dt*bf[-1 + t]*P[-1 + t]^2 - dt*P[-1 + t]^2*wf[-1 + t]^2

		# filter derivatives
		mua[t] = -3*dt*bf[-1 + t]*mu[-1 + t]^2*mua[-1 + t] + dY[t]*Pa[-1 + t]*wf[-1 + t] - dt*mu[-1 + t]*(-1 + Pa[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)) + mua[-1 + t]*(1 + dt*(af[-1 + t] - P[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)))
		mub[t] = -(dt*mu[-1 + t]^3) - 3*dt*bf[-1 + t]*mu[-1 + t]^2*mub[-1 + t] + dY[t]*Pb[-1 + t]*wf[-1 + t] - dt*mu[-1 + t]*(3*P[-1 + t] + Pb[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)) + mub[-1 + t]*(1 + dt*(af[-1 + t] - P[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)))
		muc[t] = Pc[-1 + t]*(dY[t]*wf[-1 + t] - dt*mu[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)) + muc[-1 + t]*(1 + dt*(af[-1 + t] - 3*bf[-1 + t]*(mu[-1 + t]^2 + P[-1 + t]) - P[-1 + t]*wf[-1 + t]^2))
		muw[t] = dY[t]*(P[-1 + t] + Pw[-1 + t]*wf[-1 + t]) - dt*mu[-1 + t]*(2*P[-1 + t]*wf[-1 + t] + Pw[-1 + t]*(3*bf[-1 + t] + wf[-1 + t]^2)) + muw[-1 + t]*(1 + dt*(af[-1 + t] - 3*bf[-1 + t]*(mu[-1 + t]^2 + P[-1 + t]) - P[-1 + t]*wf[-1 + t]^2))
		Pa[t] = (1 + 2*dt*af[-1 + t] - 6*dt*bf[-1 + t]*mu[-1 + t]^2)*Pa[-1 + t] - 2*dt*P[-1 + t]*(-1 + 6*bf[-1 + t]*(mu[-1 + t]*mua[-1 + t] + Pa[-1 + t]) + Pa[-1 + t]*wf[-1 + t]^2)
		Pb[t] = Pb[-1 + t] - 2*dt*(3*P[-1 + t]*(mu[-1 + t]^2 + 2*bf[-1 + t]*mu[-1 + t]*mub[-1 + t] + P[-1 + t]) - af[-1 + t]*Pb[-1 + t] + 3*bf[-1 + t]*(mu[-1 + t]^2 + 2*P[-1 + t])*Pb[-1 + t] + P[-1 + t]*Pb[-1 + t]*wf[-1 + t]^2)
		Pc[t] = 2*dt*cf[-1 + t] - 6*dt*bf[-1 + t]*(2*mu[-1 + t]*muc[-1 + t]*P[-1 + t] + (mu[-1 + t]^2 + 2*P[-1 + t])*Pc[-1 + t]) + Pc[-1 + t]*(1 + 2*dt*af[-1 + t] - 2*dt*P[-1 + t]*wf[-1 + t]^2)
		Pw[t] = -6*dt*bf[-1 + t]*(2*mu[-1 + t]*muw[-1 + t]*P[-1 + t] + (mu[-1 + t]^2 + 2*P[-1 + t])*Pw[-1 + t]) - 2*dt*P[-1 + t]^2*wf[-1 + t] + Pw[-1 + t]*(1 + 2*dt*af[-1 + t] - 2*dt*P[-1 + t]*wf[-1 + t]^2)

		# parameter estimates
		af[t] = af[t-1]+eta_a*af[t-1]*(mua[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)
		bf[t] = bf[t-1]+eta_b*bf[t-1]*(mub[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)
		cf[t] = cf[t-1]+eta_c*cf[t-1]*(muc[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)
		wf[t] = wf[t-1]+eta_w*wf[t-1]*(mu[-1 + t] + muw[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)

		# mean-squared error
		mse[t] = (X[t]-mu[t])^2
	end

	# normalize mse
	mse = mse / 1.1701
	return 0:dt:((N-1)*dt), X, dY, mu, P, af, bf, cf, wf, mse
end

end #module kalmanbucy
