using MAT
vars = matread("C:/Users/Raj/Hubei_Track.mat")

Infected = vars["Hubei_Infected_All"]
Recovered = vars["Hubei_Recovered_All"]
Dead = vars["Hubei_Dead_All"]
Time = vars["Hubei_Time"]

using Plots
using Measures
scatter(Time', Infected[1, :], xaxis = "Time(Days)", yaxis = "Wuhan - Number", label = "Data: Infected", legend = :bottomright, framestyle = :box, left_margin = 5mm)
scatter!(Time', Recovered[1, :], xaxis = "Time(Days)", yaxis = "Wuhan - Number", label = "Data: Recovered", legend = :bottomright, framestyle = :box, left_margin = 5mm)
scatter!(Time', Dead[1, :], xaxis = "Time(Days)", yaxis = "Wuhan - Number", label = "Data: Dead", legend = :bottomright, framestyle = :box, left_margin = 5mm)

function SIR(du, u, p, t)
    β, γ = p
    du[1]= dx = - β*u[1]*u[2]/u0[1]
    du[2] = dy = β*u[1]*u[2]/u0[1] - γ*u[2]
    du[3] = dz = γ*u[2]
end


u0 = Float64[11000000.0, 500 , 10]
tspan = (0.0, 39.0)
datasize = 39;
p = ([0.02, 0.013])

using DifferentialEquations
prob = ODEProblem(SIR, u0, tspan, p)
t = range(tspan[1],tspan[2],length=datasize)
sol = solve(prob, Rosenbrock23(autodiff = false),saveat=t)
using Plots
#plot(t, sol[1, :])
plot!(t, sol[1, :], linewidth = 3, color = :black, label = "SIR: Susceptible", legend = :topleft)
plot!(t, sol[2, :], linewidth = 3, color = :blue, label = "SIR: Infected", legend = :topleft)
plot!(t, sol[3, :], linewidth = 3, color = :red, label = "SIR: Recovered", title = "R0 = 11")
savefig("Wuhan_SIR_R0.pdf")

using Flux
p = param([1e-1, 1e-1])
paramsn = Flux.Params([p])


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p,saveat=t))
end

I = Infected[1, :];
R = Recovered[1,:];

function loss_adjoint()
 prediction = predict_adjoint()
 loss = sum(abs2, log.(abs.(I[1:end])) .- log.(abs.(prediction[2, :]))) + sum(abs2, log.(abs.(R[1:end]) .+ 1) .- log.(abs.(prediction[3, :] .+ 1)))
end


P1 = []
P2 = []
using Plots
using Measures
anim = Animation()
using DifferentialEquations, Flux, Optim, DiffEqFlux
using Flux
datan = Iterators.repeated((), 5000)
opt = ADAM(0.1)
cb = function()
  display(loss_adjoint())
  scatter(Time', Infected[1, :], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Infected", legend = :topleft, framestyle = :box, left_margin = 5mm)
  prediction = solve(remake(prob,p=p),Rosenbrock23(autodiff = false),saveat=t)
  display(scatter!(t, prediction[2, :], label = "Estimate- Infected"))
  scatter!(Time', Recovered[1, :], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm)
  display(scatter!(t, prediction[3, :], label = "Estimate - Recovered"))
  global P1 = append!(P1, p[1])
  global P2 = append!(P2, p[2])
  frame(anim)
end


cb()

Flux.train!(loss_adjoint, paramsn, datan, opt, cb = cb)

#=
scatter(Time', Infected[1, :], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, In, xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Prediction: Without Quarantine", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 120000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_SEIR_6.pdf")

gif(anim,"RD_Nature_NN_SIR.gif", fps=15)
=#

prediction = Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]

using JLD
save("Nature_SIR_Parameter_Corona_all_loss.jld",  "Beta_all_loss", P1 , "Gamma_all_loss", P2, "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss, "t", t)

#PLOT IN PAPER
scatter(Time', Infected[1, :], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, I_NN_all_loss, xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "SIR Prediction: Without Quarantine", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 200000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

scatter!(Time', Recovered[1, :], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, R_NN_all_loss, xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "SIR Prediction: Without Quarantine ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 200000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_8.pdf")



scatter(P1, label = "Beta")
scatter!(P2, label = "Gamma")
#=
function SEIR(du, u, p, t)
    β, μ, k, γ = p
    du[1] = - β*u[1]*u[3]/u0[1] -μ*u0[1] - μ*u[1]
    du[2] = β*u[1]*u[3]/u0[1] - (μ+k)u[2]
    du[3] = k*u[2] - (γ+ μ)*u[3]
    du[4] = γ*u[3]- μ*u[4]
end

u0 = Float64[1000000.0, 500, 500 , 0.0]
tspan = (0.0, 26.0)
datasize = 26;
p = ([1, 2e-2, 10e-2, 1e-2])

using DifferentialEquations
prob = ODEProblem(SEIR, u0, tspan, p)
t = range(tspan[1],tspan[2],length=datasize)
sol = solve(prob, Rosenbrock23(autodiff = false),saveat=t)
using Plots
#plot(t, sol[1, :])
plot(t, sol[2, :])
plot!(t, sol[4, :])
=#
