include("src/bimodal.jl")
using .bimodal

Ntrials = 100
T = 2000
dt = 0.001
par = [4., 3., 1., 2.]
decpar = [1., 2., 3., 4.]
rates = [1e-2, 1e-2, 4e-3, 1e-2]
window = 20000 #window for moving averages



#----------------------------------------------------------------------
# Run simulation
#----------------------------------------------------------------------
t,X,dY,mu,P,af,bf,cf,wf,mse = path(T,dt,par,decpar,rates)



#----------------------------------------------------------------------
# Produce figure
#----------------------------------------------------------------------
using Plots
fig4a = plot(;
  xlabel = "time (in units of hidden state time-constant)",
  grid = false,
  ylimit = (-2.5, 2.5),
  size = (1000, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig4a,
  t[begin:100:end], X[begin:100:end];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig4a,
  t[begin:100:end], mu[begin:100:end];
  color = :red, lw = 1,
  label = "state estimate"
)

fig4b = plot(;
  grid = false,
  ylimit = (-2.5, 2.5),
  size = (500, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig4b,
  t[begin:100:100000], X[begin:100:100000];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig4b,
  t[begin:100:100000], mu[begin:100:100000];
  color = :red, lw = 1,
  label = "state estimate"
)
plot!(
  fig4b,
  t[begin:100:100000],
  mu[begin:100:100000] - sqrt.(P[begin:100:100000]);
  fill = mu[begin:100:100000] + sqrt.(P[begin:100:100000]),
  color = :red, lw = 1, opacity = 0.3,
  label = nothing
)

fig4c = plot(;
  grid = false,
  ylimit = (-2.5, 2.5),
  size = (500, 300),
  bottommargin = 1Plots.cm,
  legend = nothing
)
plot!(
  fig4c,
  t[end-100000:100:end], X[end-100000:100:end];
  color = :black, lw = 1,
  label = "hidden state"
)
plot!(
  fig4c,
  t[end-100000:100:end], mu[end-100000:100:end];
  color = :red, lw = 1,
  label = "state estimate"
)
plot!(
  fig4c,
  t[end-100000:100:end],
  mu[end-100000:100:end] - sqrt.(P[end-100000:100:end]);
  fill = mu[end-100000:100:end] + sqrt.(P[end-100000:100:end]),
  color = :red, lw = 1, opacity = 0.3,
  label = nothing
)

fig4l = plot(; axis = false, grid = false, size = (200, 300))
plot!(fig4l, []; lw = 2, color = :black, label = " hidden state")
plot!(fig4l, []; lw = 2, color = :red, label = " state estimate")

fig4bottom = plot(
  fig4b, fig4l, fig4c;
  size = (1000, 200),
  layout = grid(1, 3, widths = [0.45, 0.1, 0.45])
)

fig4 = plot(
  fig4a, fig4bottom;
  layout = (2, 1),
  size = (1000, 500)
)

savefig(fig4, "fig/fig4.png")
