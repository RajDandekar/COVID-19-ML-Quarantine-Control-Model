#This file was used to generate the .JLD file used in the generate plots code

Pkg.add("MAT")
using MAT
vars = matread("C:/Users/Raj/US_Track_Dead.mat")

Infected = vars["US_Infected_All"]
Recovered = vars["US_Recovered_All"]
Dead = vars["US_Dead_All"]
Time = vars["US_Time"]

Pkg.add("Plots")
Pkg.add("Measures")

using Plots
using Measures
scatter(Time, Infected, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Infected", legend = :bottomright, framestyle = :box, left_margin = 5mm)
scatter!(Time, Recovered, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Recovered", legend = :bottomright, framestyle = :box, left_margin = 5mm)
scatter!(Time, Dead, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Dead", legend = :bottomright, framestyle = :box, left_margin = 5mm)
#=
function QSIR(du, u, p, t)
    β = 0.2345
    γ = 0.01193
    du[1]= dx = - β*u[1]*(u[2] - 10000*p[1])/u0[1]
    du[2] = dy = (β*u[1]*(u[2] - 10000*p[1])/u0[1] - γ*u[2])
    du[3] = dz = γ*u[2]
end=#

Pkg.add("Flux")
using Flux
ann = Chain(Dense(4,10,relu), Dense(10,1))
p1,re = Flux.destructure(ann)
p2 = Float64[0.5, 0.03]
p3 = [p1; p2]
ps = Flux.params(p3)


function QSIR(du, u, p, t)
    β = abs(p[62])
    γ = 0.007627
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - abs(re(p[1:61])(u)[1])*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  abs(re(p[1:61])(u)[1]*u[2]/u0[1])
end


u0 = Float64[327000000.0, 518 ,7, 10]
tspan = (0, 24.0)
datasize = 24;
#p = param([0.18])
#p = ([0.2, 0.013, 2])

Pkg.add("DifferentialEquations")
using DifferentialEquations
prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

sol = Array(concrete_solve(prob, Rosenbrock23(autodiff = false),u0, p3, saveat=t))

#sol = solve(prob, Rosenbrock23(autodiff = false),saveat=Time)

using Plots
#plot(t, sol[1, :])
#plot!(t, sol[1, :], linewidth = 3, color = :black, label = "SIR: Susceptible", legend = :topleft)
plot!(t, sol[2, :], linewidth = 3, color = :blue, label = "SIR: Infected", legend = :topleft)
plot!(t, sol[3, :], linewidth = 3, color = :red, label = "SIR: Recovered", title = "R0 = 11")

#=using Flux
p = param([ 0.5])
paramsn = Flux.Params([p])=#


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p3,saveat=t))
end

I = Infected[1, :];
R = Recovered[1,:];

function loss_adjoint()
 prediction = predict_adjoint()
 loss = sum(abs2, log.(abs.(Infected)) .- log.(abs.(prediction[2, :])))
end


Loss = []
P1 = []
P2 = []
P3 = []

#P3  =[]
anim = Animation()
#Pkg.add("Optim")
#Pkg.add("DiffEqFlux")
using DiffEqFlux
datan = Iterators.repeated((), 2500)
opt = ADAM(0.01)
cb = function()
  display(loss_adjoint())
  scatter(Time, Infected, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm)
  prediction = solve(remake(prob,p=p3),Rosenbrock23(autodiff = false),saveat=Time)
  display(scatter!(t, prediction[2, :], label = "NN - Infected"))
  scatter!(Time, Recovered + Dead, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm)
  display(scatter!(t, prediction[3, :], label = "NN - Recovered"))
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[62])
  global P2 = append!(P2, p3[63])
  global P3 = append!(P3, p3)
  frame(anim)
end


cb()

Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

gif(anim,"Dead_USn.gif", fps=15)

using JLD
D = load("US_31March_QSIR_Deadn.jld")
p3 = D["Parameters"]


prediction = Array(concrete_solve(prob,Rosenbrock23(autodiff = false),u0,p3,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]
T_NN_all_loss = prediction[4, :]

 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(re(p3[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1])
end

Pkg.add("LaTeXStrings")
using LaTeXStrings

#Starting 8 March 2020
scatter(Time, Infected, xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, prediction[2, :], xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time, Recovered + Dead, xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, prediction[3, :],  xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (-5000, 200000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("US_1dn.pdf")

scatter(t,Q_parameter/u0[1],  ylims = (0.3, 0.7), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("US_2dn.pdf")

using LaTeXStrings
scatter(t, abs(p3[62]) ./ (0.007627 .+ Q_parameter/u0[1]), ylims = (0.5, 2), xlabel = "Days post 500 infected", ylabel = L"R_{t}", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = L"R_{t} = 1")
savefig("US_3dn.pdf")

#savefig("Nature_US_New_Final_Effective_Reproduction_Number_Time.pdf")

#scatter(p3[52]/p3[53], xlabel = "Time(Days)", ylabel = "Base Reproduction Ratio", label = "US Data", legend = :bottomright)
#savefig("Final_Base_Reproduction_Number_Time.pdf")

#Iter = range(500, length(P1), length =36)

#I DID NOT PLOT THESE FOR THE LATEST VERSION OF THE PAPER

#=
scatter(Iter, P1[500:50:end], xlabel = "Number of iterations", label = L"$\beta$", color = :black, framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0.4, 0.6),  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_4n.pdf")

#savefig("Nature_US_New_Final_NN_Quarantine_Beta_Iterations.pdf")

scatter(Iter, P2[500:50:end], xlabel = "Number of iterations", label = L"$\gamma$", color = :black, framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0, 0.1),  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_7.pdf")

Iter_new = range(400, length(Loss)-800, length =36)
scatter(Iter_new,Loss[400:30:end-800], xlabel = "Number of iterations", ylabel = "Loss", label = "Neural network loss", color = :black, framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, right_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0, 40), xlims = (350, 1500) ,yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("RD_Nature_QSIR_5.pdf")
=#


Pkg.add("JLD")
using JLD
save("US_31March_QSIR_Deadn.jld",  "Q_parameter", Q_parameter , "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss, "T_NN_all_loss", T_NN_all_loss, "t", t, "Parameters", p3, "Loss", Loss, "P1", P1, "P2", P2)

#Quarantined population prediction
#=
Q_pop = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
 Q_pop[i] = re(p3[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1] * I_NN_all_loss[i]/ u0[1]
end

scatter(t,Q_pop, xlabel = "Days since 8 March 2020", ylabel = "US: number of cases", label = "Quarantined: Prediction",color = :green, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_14n.pdf")
=#

#savefig("Nature_US_New_Final_NN_Quarantine_Loss_Iterations.pdf")


#for forecasting


using JLD
D = load("US_QSIR.jld")
p3n = D["Parameters"]


p3n = p3
tspan_n = (0, 50.0)
datasize_n = 50;
t_n = range(tspan_n[1],tspan_n[2],length=datasize_n)
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))

scatter(Time, Infected, yaxis = "US: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t_n, predictionn[2, :], xticks = ([0:25:50;], ["8 March' 20", " 2 April' 20", "27 April' 20"]),  yaxis = "US: Number of cases", label = "Infected forecast", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time, Recovered, yaxis = "US: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t_n, predictionn[3, :], xticks = ([0:25:50;], ["8 March' 20", "2 April' 20", "27 April' 20"]),  yaxis = "US: Number of cases", label = "Recovered forecast ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm,right_margin = 8mm,grid = :off, color = :blue, linewidth  = 3, ylims = (0, 600000), foreground_color_legend = nothing,background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)


savefig("US_4dn.pdf")

#New plots of uarantine population

S_NN_all_loss = predictionn[1, :]
I_NN_all_loss = predictionn[2, :]
R_NN_all_loss = predictionn[3, :]
T_NN_all_loss = predictionn[4, :]


 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

 for i = 1:length(S_NN_all_loss)
   Q_parameter[i] = abs(re(p3n[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1])
 end

 scatter(t_n[25:50],Q_parameter[25:50]/u0[1], ylims = (0.5, 0.8),  ylabel = "Q(t)",  xticks = ([25:10:45;], ["25", "35", "45"]), xlabel = "Days post 500 infected"
 , label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, right_margin = 8mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("US_5dn.pdf")

using LaTeXStrings
scatter(t_n[25:50], abs(p3n[62]) ./ (0.007627.+ Q_parameter[25:50]/u0[1]), ylims = (0.5, 1.5), xticks = ([25:10:45;], ["25", "35", "45"]), xlabel = "Days post 500 infected"
,  ylabel = L"R_{t}", label = "Effective reproduction number", right_margin = 8mm, legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = L"R_{t} = 1")


savefig("US_6dn.pdf")
#=
using JLD
save("US_Nature_QSIR.jld",  "Q_parameter", Q_parameter , "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss, "t", t, "Parameters", p3, "Loss", Loss, "P1", P1, "P2", P2)
=#

#=D = load("Split_Parameter_Corona_all_loss.jld")

T_1= D["t"]
I_1= D["I_NN_all_loss"]
R_1= D["R_NN_all_loss"]
Beta = D["Beta_all_loss"]
Gamma = D["Gamma_all_loss"]=#



#=scatter(Time', Infected[1, :], xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Infected", legend = :bottomright, framestyle = :box, left_margin = 5mm)
scatter!(Time', Recovered[1, :], xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Recovered", legend = :bottomright, framestyle = :box, left_margin = 5mm)
plot!(T_1, I_1, label = "SIR - Infected", color = :black, linewidth = 3)
plot!(T_1, R_1, label = "SIR - Recovered", color = :black, linewidth = 3)



D = load("Split_Quarantine_Parameter_Corona_all_loss.jld")

T_2= D["t"]
I_2= D["I_NN_all_loss"]
R_2= D["R_NN_all_loss"]
Q = D["Quarantine_all_loss"]

plot!(T_2, I_2, label = "Quarantine SIR - Infected", color = :purple, linewidth = 3)
plot!(T_2, R_2, label = "Quarantine SIR - Recovered", color = :purple, linewidth = 3, legend = :topleft)
savefig("Split_Quarantine.pdf")

scatter(Beta, label = "Beta")
scatter!(Gamma, label = "Gamma")
scatter(Q, label = "Quarantine Parameter = Q")
savefig("Split_Parameters_Q.pdf")=#


#for forecasting-With other countries data

D = load("Mac_RD_Nature_QSIR.jld")
p3n = D["Parameters"]


#Using US NN till Day 10 and then using Wuhan's NN
u0 = Float64[327000000.0, 7436 ,40, 10]
tspan_n = (10.0, 40.0)
datasize_n = 30;
t_n = range(tspan_n[1],tspan_n[2],length=datasize_n)
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))

scatter(Time, Infected, yaxis = "US: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t_n, predictionn[2, :], xticks = ([10:15:40;], ["18 March' 20", " 2 April' 20", "17 April' 20"]),  yaxis = "US: Number of cases", label = "Infected forecast", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time, Recovered, yaxis = "US: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t_n, predictionn[3, :], xticks = ([10:15:30;], ["18 March' 20", "2 April' 20", "17 April' 20"]),  yaxis = "US: Number of cases", label = "Recovered forecast ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm,right_margin = 8mm,grid = :off, color = :blue, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing,background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
