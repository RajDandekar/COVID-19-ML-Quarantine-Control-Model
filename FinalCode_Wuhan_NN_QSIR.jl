#This file was used to generate the .JLD file used in the generate plots code

using MAT
using Plots
using Measures
using Flux
using DifferentialEquations
using DiffEqFlux
using LaTeXStrings
vars = matread("C:/Users/Raj/Hubei_Track.mat")

Infected = vars["Hubei_Infected_All"]
Recovered = vars["Hubei_Recovered_All"]
Dead = vars["Hubei_Dead_All"]
Time = vars["Hubei_Time"]

ann = Chain(Dense(4,10,relu), Dense(10,1))
p1,re = Flux.destructure(ann)
p2 = Float64[0.02, 0.01]
p3 = [p1; p2]
ps = Flux.params(p3)


function QSIR(du, u, p, t)
    β = abs(p[62])
    γ = abs(p[63])
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - abs(re(p[1:61])(u)[1])*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  + abs(re(p[1:61])(u)[1])*u[2]/u0[1]
end


u0 = Float64[11000000.0, 500 ,10, 10]
tspan = (0, 39.0)
datasize = 39;

prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

sol = Array(concrete_solve(prob, Rosenbrock23(autodiff = false),u0, p3, saveat=t))


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p3,saveat=t))
end

I = Infected[1, :];
R = Recovered[1,:];
D = Dead[1, :];

function loss_adjoint()
 prediction = predict_adjoint()
 loss = sum(abs2, log.(abs.(I[1:39])) .- log.(abs.(prediction[2, :]))) + sum(abs2, log.(abs.(R[1:39] + D[1:39]) .+ 1) .- log.(abs.(prediction[3, :] .+ 1)))
end


Loss = []
P1 = []
P2 = []
P3 = []

#P3  =[]
anim = Animation()
using DiffEqFlux
datan = Iterators.repeated((), 500)
opt = ADAM(0.1)
cb = function()
  display(loss_adjoint())
  scatter(Time[1:39], I[1:39], xaxis = "Time(Days)", yaxis = "Wuhan - Number", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm)
  prediction = solve(remake(prob,p=p3),Rosenbrock23(autodiff = false),saveat=Time)
  display(scatter!(t, prediction[2, :], label = "NN - Infected"))
  scatter!(Time[1:39], R[1:39] + D[1:39], xaxis = "Time(Days)", yaxis = "Wuhan - Number", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm)
  display(scatter!(t, prediction[3, :], label = "NN - Recovered"))
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[62])
  global P2 = append!(P2, p3[63])
  global P3 = append!(P3, p3)
  frame(anim)
end


cb()

#STOP ITERATIONS WHEN LOSS FUNCTION STARTS TO STAGNATE
Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

gif(anim,"Dead_Wuhan.gif", fps=15)

L = findmin(Loss)
idx = L[2]
idx1 = (idx-1)*63 +1
idx2 = idx*63
p3 = P3[idx1: idx2]

prediction = Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p3,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]
T_NN_all_loss = prediction[4, :]

 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(re(p3[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1])
end

#INFECTED AND RECOVERED COUNT
scatter(Time[1:39], I[1:39], xaxis = "Days post 500 infected", yaxis = "Wuhan: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, prediction[2, :], xaxis = "Days post 500 infected", yaxis = "Wuhan: Number of cases", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
scatter!(Time[1:39], R[1:39] +  D[1:39], xaxis = "Days post 500 infected", yaxis = "Wuhan: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, prediction[3, :], xaxis = "Days post 500 infected", yaxis = "Wuhan: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("Wuhan_1d.pdf")

#QUARANTINE STRENGTH
scatter(t,Q_parameter/u0[1], xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0.7, 1.2),  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("Wuhan_2d.pdf")

#REPRODUCTION NUMBER
scatter(t, abs(p3[62]) ./ (abs(p3[63]) .+ Q_parameter/u0[1]), xlabel = "Days post 500 infected", ylabel = L"R_{t}", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0.5, 2), yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = "R = 1")
savefig("Wuhan_3d.pdf")
