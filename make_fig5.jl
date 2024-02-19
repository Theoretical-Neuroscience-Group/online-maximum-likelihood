include("src/bimodal.jl")
using .bimodal
using Statistics

Ntrials = 100
T = 2000
dt = 0.001
N = ceil(Int, T / dt)
par = [4., 3., 1., 2.]
decpar = [1., 2., 3., 4.]
rates = [1e-2, 1e-2, 4e-3, 1e-2]
window = Int64(20000) #window for moving averages



#----------------------------------------------------------------------
# Run simulation without learning
#----------------------------------------------------------------------
println("WITHOUT LEARNING")

#multiple trials
avgmse=zeros(N)

println("running simulations...")
for i=1:Ntrials
  println(i)
  t,X,dY,mu,P,af,bf,cf,wf,mse = path(T,dt,par,decpar,0*rates)
  avgmse=avgmse+mse
end
mse=avgmse/Ntrials


#moving average
println("computing moving average...")
mvavg=zeros(N)
mvavg[1]=mean(mse[1:window])
@time for i=2:(N-window)
  mvavg[i]=mvavg[i-1]-mse[i-1]/window+mse[i+window-1]/window
end
for i=(N-window+1):N-1
#   println(i)
   mvavg[i]=((N-i+1)*mvavg[i-1]-mse[i-1])/(N-i)
end

mvavg1=copy(mvavg)



#----------------------------------------------------------------------
# Run simulation with learning
#----------------------------------------------------------------------
println("WITH LEARNING")

#multiple trials
avgmse=zeros(N)
avgaf=zeros(Int64(T))
avgbf=zeros(Int64(T))
avgcf=zeros(Int64(T))
avgwf=zeros(Int64(T))

println("running simulations...")
for i=1:Ntrials
  println(i)
  t,X,dY,mu,P,af,bf,cf,wf,mse = path(T,dt,par,decpar,rates)
  avgmse=avgmse+mse
  avgaf=avgaf+af[1000:1000:N]
  avgbf=avgbf+bf[1000:1000:N]
  avgcf=avgcf+cf[1000:1000:N]
  avgwf=avgwf+wf[1000:1000:N]
end
mse=avgmse/Ntrials
avgaf=avgaf/Ntrials
avgbf=avgbf/Ntrials
avgcf=avgcf/Ntrials
avgwf=avgwf/Ntrials


#moving average
println("computing moving average...")
mvavg=zeros(N)
mvavg[1]=mean(mse[1:window])
@time for i=2:(N-window)
  mvavg[i]=mvavg[i-1]-mse[i-1]/window+mse[i+window-1]/window
end
for i=(N-window+1):N-1
#   println(i)
   mvavg[i]=((N-i+1)*mvavg[i-1]-mse[i-1])/(N-i)
end

mvavg2=copy(mvavg)



#----------------------------------------------------------------------
# Run simulation with ground truth
#----------------------------------------------------------------------
println("GROUND TRUTH")

#multiple trials
N=Int64(cld(T,dt))
avgmse=zeros(N)

for i=1:Ntrials
  println(i)
  t,X,dY,mu,P,af,bf,cf,wf,mse = path(T,dt,par,par,0*rates)
  avgmse=avgmse+mse
end
mse=avgmse/Ntrials

#moving average
mvavg=zeros(N)

println("computing moving average...")
mvavg[1]=mean(mse[1:window])
@time for i=2:(N-window)
  mvavg[i]=mvavg[i-1]-mse[i-1]/window+mse[i+window-1]/window
end
for i=(N-window+1):N-1
#   println(i)
   mvavg[i]=((N-i+1)*mvavg[i-1]-mse[i-1])/(N-i)
end

mvavg3=copy(mvavg)



#----------------------------------------------------------------------
# Produce figure
#----------------------------------------------------------------------
using Plots
fig5a = plot(;
  ylabel = "normalized mean squared error",
  xlabel = "time (in units of hidden state time-constant)",
  grid = false,
  ylimit = (0, 1)
)
plot!(
  fig5a,
  0:10dt:(N-window)*dt, mvavg1[1:10:end-window+1];
  color = :magenta, lw = 2,
  label = "no learning"
)
plot!(
  fig5a,
  0:10dt:(N-window)*dt, mvavg2[1:10:end-window+1];
  color = :blue, lw = 2,
  label = "learning"
)
plot!(
  fig5a,
  0:10dt:(N-window)*dt, mvavg3[1:10:end-window+1];
  color = :darkgray, lw = 2,
  label = "ground truth"
)

fig5b = plot(;
  ylabel = "parameter estimates",
  xlabel = "time (in units of hidden state time-constant)",
  grid = false,
  # ylimit = (0, 1.2)
)
plot!(
  fig5b,
  avgaf;
  color = :blue, lw = 2,
  label = "ãₜ"
)
plot!(
  fig5b,
  avgbf;
  color = :blue, lw = 2, ls = :dash,
  label = "b̃ₜ"
)
plot!(
  fig5b,
  avgcf;
  color = :blue, lw = 2, ls = :dashdot,
  label = "σ̃ₜ"
)
plot!(
  fig5b,
  avgwf;
  color = :blue, lw = 2, ls = :dot,
  label = "w̃ₜ"
)
hline!(fig5b, [par[1]]; color = :darkgray, lw = 2, label = "a₀")
hline!(fig5b, [par[2]]; color = :darkgray, lw = 2, label = "b₀", ls = :dash)
hline!(fig5b, [par[3]]; color = :darkgray, lw = 2, label = "σ₀", ls = :dash)
hline!(fig5b, [par[4]]; color = :darkgray, lw = 2, label = "w₀", ls = :dot)

fig5 = plot(
  fig5a, fig5b;
  layout = (1, 2),
  size = (1200, 400),
  margin = 1Plots.cm
)
savefig(fig5, "fig/fig5.png")
