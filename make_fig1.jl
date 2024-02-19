include("src/kalmanbucy.jl")
using .kalmanbucy

Ntrials = 100
T = 1000
dt = 0.001
par = [1., 2., 3.]
decpar = [10., sqrt(0.2), 3.]
rates = [0.03, 0.03, 0.0]
window = 20000 #window for moving averages



#----------------------------------------------------------------------
# Run simulation
#----------------------------------------------------------------------
t,X,dY,mu,P,af,bf,wf,mse = path(T,dt,par,decpar,rates)



#----------------------------------------------------------------------
# Produce figure
#----------------------------------------------------------------------
using Plots
fig1a = plot(;
  xlabel = "time (in units of hidden state time-constant)",
  grid = false,
  ylimit = (-5, 5),
  size = (1000, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig1a,
  t[begin:100:end], X[begin:100:end];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig1a,
  t[begin:100:end], mu[begin:100:end];
  color = :red, lw = 1,
  label = "state estimate"
)

fig1b = plot(;
  grid = false,
  ylimit = (-5, 5),
  size = (500, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig1b,
  t[begin:100:10000], X[begin:100:10000];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig1b,
  t[begin:100:10000], mu[begin:100:10000];
  color = :red, lw = 1,
  label = "state estimate"
)
plot!(
  fig1b,
  t[begin:100:10000],
  mu[begin:100:10000] - sqrt.(P[begin:100:10000]);
  fill = mu[begin:100:10000] + sqrt.(P[begin:100:10000]),
  color = :red, lw = 1, opacity = 0.3,
  label = nothing
)

fig1c = plot(;
  grid = false,
  ylimit = (-5, 5),
  size = (500, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig1c,
  t[end-10000:100:end], X[end-10000:100:end];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig1c,
  t[end-10000:100:end], mu[end-10000:100:end];
  color = :red, lw = 1,
  label = "state estimate"
)
plot!(
  fig1c,
  t[end-10000:100:end],
  mu[end-10000:100:end] - sqrt.(P[end-10000:100:end]);
  fill = mu[end-10000:100:end] + sqrt.(P[end-10000:100:end]),
  color = :red, lw = 1, opacity = 0.3,
  label = nothing
)

fig1l = plot(; axis = false, grid = false, size = (200, 300))
plot!(fig1l, []; lw = 2, color = :black, label = " hidden state")
plot!(fig1l, []; lw = 2, color = :red, label = " state estimate")

fig1bottom = plot(
  fig1b, fig1l, fig1c;
  size = (1000, 200),
  layout = grid(1, 3, widths = [0.45, 0.1, 0.45])
)

fig1 = plot(
  fig1a, fig1bottom;
  layout = (2, 1),
  size = (1000, 500)
)

savefig(fig1, "fig/fig1.png")
