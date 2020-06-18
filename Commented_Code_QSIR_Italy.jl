
####COVID MODELLING CODE USING JULIA LANGUAGE####

#In Julia, you need to load the major packages before running the code.
#First you have to add  the packages if not already added through Pkg.add(""). This needs to be done just once.
#Then you need to do using "Pkg" each time you run the code as done below

#Package for loading Matlab files
using MAT

#Plotting packages
using Plots
using Measures


#ODE solving + ML integration packages
using Flux
using DifferentialEquations
using DiffEqFlux

#Latex display  packages
using LaTeXStrings

#For seed generator
Random.seed!(50)

#For saving data
using JLD

#Load Matlab variables
vars = matread("C:/Users/Raj/Italy_Track.mat")

Infected = vars["Italy_Infected_All"]
Recovered = vars["Italy_Recovered_All"]
Dead = vars["Italy_Dead_All"]
Time = vars["Italy_Time"]

#Define neural network architecture
#Very flexible: You can play around with units, layers and choice of activation function.

#The following has one hidden layer with 10 units and relu output
ann = Chain(Dense(4,10,relu), Dense(10,1))

#To have 2 hidden layers and 20 units in each layer, the code just changes to
#ann = Chain(Dense(4,20,relu), Dense(20, 20, relu), Dense(20,1))

#Get parameters of the neural network(weights)
p1,re = Flux.destructure(ann)

#Initial guesses of β, γ
p2 = Float64[0.1, 0.03]

#Get the augmented parameter matrix. Note: there will be a total of 63 parameters.
p3 = [p1; p2]
ps = Flux.params(p3)

#QSIR ODE with the neural network term for the quarantined population
#Can play around with the number of compartments in the SIR model, number of neural networks.
#Can even work with PDE's if you want to add spatial component to your model
function QSIR(du, u, p, t)
    β = abs(p[62])
    γ = abs(p[63])
    NN1 = abs(re(p[1:61])(u)[1])
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - NN1*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  NN1*u[2]/u0[1]
end

#Initialization: Susceptibles is the initial population, here 60 million for Italy. Infected and Recovered are initialized from data. Quarantined is assumed to be very small ~ 10.
u0 = Float64[60000000.0, 650 ,45, 10]
tspan = (0, 27.0)
datasize = 27;


prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

#There are a number of solvers Julia provides. I've used Tsit5() since its the fastest.
#If your system of ODE becomes stiff, consider using Rosenbrock(), disadvantage being that its slower.
sol = Array(concrete_solve(prob, Tsit5(),u0, p3, saveat=t))



function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=t))
end

I = Infected[1, :];
R = Recovered[1,:];

#Loss function based on te infected and recovered data. Note: recovered = healthy recovered + dead (basically total population which cannot further infect the susceptibles.)
function loss_adjoint()
 prediction = predict_adjoint()
 loss = sum(abs2, log.(abs.(Infected)) .- log.(abs.(prediction[2, :]))) + sum(abs2, log.(abs.(Recovered + Dead) .+ 1) .- log.(abs.(prediction[3, :] .+ 1)))
end


#Store parameters while iterations
Loss = []
P1 = []
P2 = []
P3 = []

#P3  =[]
anim = Animation()
#Here, iterations = 2500. You can play around with this number.
datan = Iterators.repeated((), 2000)
opt = ADAM(0.1)
cb = function()
  #Display loss while running iterations.
  display(loss_adjoint())
  #Display animation while running iterations. Comment this off if you want speed.
  scatter(Time, Infected, xaxis = "Time(Days)", yaxis = "Italy - Number", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm)
  prediction = solve(remake(prob,p=p3),Tsit5(),saveat=Time)
  display(scatter!(t, prediction[2, :], label = "NN - Infected"))
  scatter!(Time, Recovered + Dead, xaxis = "Time(Days)", yaxis = "Italy - Number", label = "Data: Recovered + Dead", legend = :topleft, framestyle = :box, left_margin = 5mm)
  display(scatter!(t, prediction[3, :], label = "NN - Recovered"))
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[62])
  global P2 = append!(P2, p3[63])
  global P3 = append!(P3, p3)
  frame(anim)
end


cb()

#Training of the neural network. Optimize all the parameters:
Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

#Save animation if need be
gif(anim,"Dead_Italy.gif", fps=15)

L = findmin(Loss)
idx = L[2]
idx1 = (idx-1)*63 +1
idx2 = idx*63
p3 = P3[idx1: idx2]

#Now plot the data
prediction = Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]
T_NN_all_loss = prediction[4, :]

 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(re(p3[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1])
end


#Infected and recovered count
scatter(Time, Infected, xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, prediction[2, :], xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
scatter!(Time, Recovered + Dead, xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, prediction[3, :], xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 100000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("Italy_1dn.pdf")

#Quarantine strength
scatter(t,Q_parameter/u0[1], ylims = (0.3, 1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("Italy_2dn.pdf")

#Reproduction number
scatter(t, abs(p3[62]) ./ (abs(p3[63]) .+ Q_parameter/u0[1]), ylims= (0.5, 2),  xlabel = "Days post 500 infected", ylabel = L"R_{t}", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = L"R_{t} = 1")
savefig("Italy_3dn.pdf")

#Save important data
save("QSIR_Italy.jld", "Q_parameter", Q_parameter, "prediction", prediction,  "β_parameter", p3[62],"γ_parameter", p3[63], "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss, "T_NN_all_loss", T_NN_all_loss, "t", t, "Parameters", p3, "Loss", Loss)
