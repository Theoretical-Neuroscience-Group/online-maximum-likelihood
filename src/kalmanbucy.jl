module kalmanbucy

export path

function path(T,dt,par,decpar,rates)
	# number of time-steps
	N = ceil(Int, T / dt)

	# parameters
	a = par[1]
	b = par[2]
	w = par[3]

	#l earning rates
	eta_a = rates[1]
	eta_b = rates[2]
	eta_w = rates[3]

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
	muw = zeros(N)
	Pa = zeros(N)
	Pb = zeros(N)
	Pw = zeros(N)

	# parameter estimates
	af = zeros(N)
	bf = zeros(N)
	wf = zeros(N)

	# initialize parameter estimates
	af[1] = decpar[1]
	bf[1] = decpar[2]
	wf[1] = decpar[3]

	# special initial conditions
	X[1] = randn()
	P[1] = bf[1]^2/(2*af[1])
	Pa[1] = -bf[1]^2/(2*af[1]^2)
	Pb[1] = bf[1]/af[1]
	Pw[1] = 0

	# mean-squared error
	mse = zeros(N)
	mse[1] = X[1]^2

	#Euler-Maruyama integration of stochastic variables
	for t = 2:N
		# state and observation
		X[t] = b*dW[t] + X[-1 + t] - a*dt*X[-1 + t]
		dY[t]= dV[t] + dt*w*X[-1 + t]

		# filter
		mu[t] = mu[-1 + t] - dt*af[-1 + t]*mu[-1 + t] + dY[t]*P[-1 + t]*wf[-1 + t] - dt*mu[-1 + t]*P[-1 + t]*wf[-1 + t]^2
		P[t] = dt*bf[-1 + t]^2 + P[-1 + t] - 2*dt*af[-1 + t]*P[-1 + t] - dt*P[-1 + t]^2*wf[-1 + t]^2

		# filter derivatives
		mua[t] = dY[t]*Pa[-1 + t]*wf[-1 + t] - mua[-1 + t]*(-1 + dt*af[-1 + t] + dt*P[-1 + t]*wf[-1 + t]^2) - mu[-1 + t]*(dt + dt*Pa[-1 + t]*wf[-1 + t]^2)
		mub[t] = Pb[-1 + t]*wf[-1 + t]*(dY[t] - dt*mu[-1 + t]*wf[-1 + t]) - mub[-1 + t]*(-1 + dt*af[-1 + t] + dt*P[-1 + t]*wf[-1 + t]^2)
		muw[t] = dY[t]*P[-1 + t] - muw[-1 + t]*(-1 + dt*af[-1 + t] + dt*P[-1 + t]*wf[-1 + t]^2) + wf[-1 + t]*(dY[t]*Pw[-1 + t] - dt*mu[-1 + t]*(2*P[-1 + t] + Pw[-1 + t]*wf[-1 + t]))
		Pa[t] = (1 - 2*dt*af[-1 + t])*Pa[-1 + t] - 2*P[-1 + t]*(dt + dt*Pa[-1 + t]*wf[-1 + t]^2)
		Pb[t] = 2*dt*bf[-1 + t] + Pb[-1 + t]*(1 - 2*dt*(af[-1 + t] + P[-1 + t]*wf[-1 + t]^2))
		Pw[t] = -2*dt*P[-1 + t]^2*wf[-1 + t] + Pw[-1 + t]*(1 - 2*dt*(af[-1 + t] + P[-1 + t]*wf[-1 + t]^2))

		# parameter estimates
		af[t] = af[t-1]+eta_a*af[t-1]*(mua[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)
		bf[t] = bf[t-1]+eta_b*bf[t-1]*(mub[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)
		wf[t] = wf[t-1]+eta_w*wf[t-1]*(mu[-1 + t] + muw[-1 + t]*wf[-1 + t])*(dY[t]-(mu[-1 + t]*wf[-1 + t])dt)

		# mean-squared error
		mse[t]=(X[t]-mu[t])^2
	end

	#renormalize mse
	mse = 2 * a * mse / b^2
	return 0:dt:((N-1)*dt), X, dY, mu, P, af, bf, wf, mse
end

end #module kalmanbucy
