#This file was used to generate the .JLD file used in the generate plots code

Pkg.add("MAT")
using MAT
vars = matread("C:/Users/Raj/US_Track.mat")

Infected = vars["US_Infected_All"]
Recovered = vars["US_Recovered_All"]
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
    γ = abs(p[63])
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - abs(re(p[1:61])(u)[1])*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  abs(re(p[1:61])(u)[1]*u[2]/u0[1])
end


u0 = Float64[327000000.0, 518 ,7, 10]
tspan = (0, 17.0)
datasize = 17;
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
 loss = sum(abs2, log.(abs.(Infected)) .- log.(abs.(prediction[2, :]))) + sum(abs2, log.(abs.(Recovered) .+ 1) .- log.(abs.(prediction[3, :] .+ 1)))
end


Loss = []
P1 = []
P2 = []
P3 = []

#P3  =[]
anim = Animation()
Pkg.add("Optim")
Pkg.add("DiffEqFlux")
using DifferentialEquations, Flux, Optim, DiffEqFlux
using Flux
datan = Iterators.repeated((), 2500)
opt = ADAM(0.01)
cb = function()
  display(loss_adjoint())
  scatter(Time, Infected, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm)
  prediction = solve(remake(prob,p=p3),Rosenbrock23(autodiff = false),saveat=Time)
  display(scatter!(t, prediction[2, :], label = "NN - Infected"))
  scatter!(Time, Recovered, xaxis = "Time(Days)", yaxis = "US - Number", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm)
  display(scatter!(t, prediction[3, :], label = "NN - Recovered"))
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[62])
  global P2 = append!(P2, p3[63])
  global P3 = append!(P3, p3)
  frame(anim)
end


cb()

Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

gif(anim,"US_Nature_Quarantine_NN_SIR_full.gif", fps=15)

using JLD
D = load("US_QSIR.jld")
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

scatter!(Time, Recovered, xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, prediction[3, :], xaxis = "Days post 500 infected", yaxis = "US: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("US_1.pdf")

scatter(t,Q_parameter/u0[1], ylims = (-0.05, 0.5), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("US_2.pdf")


scatter(t, abs(p3[62]) ./ (abs(p3[63]) .+ Q_parameter/u0[1]),  xlabel = "Days post 500 infected", ylabel = "R(t)", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = "R = 1")
savefig("US_3.pdf")

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
save("US_QSIR.jld",  "Q_parameter", Q_parameter , "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss, "T_NN_all_loss", T_NN_all_loss, "t", t, "Parameters", p3, "Loss", Loss, "P1", P1, "P2", P2)

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
tspan_n = (0, 30.0)
datasize_n = 30;
t_n = range(tspan_n[1],tspan_n[2],length=datasize_n)
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))

scatter(Time, Infected, yaxis = "US: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t_n, predictionn[2, :], xticks = ([0:15:30;], ["8 March' 20", " 23 March' 20", "7 April' 20"]),  yaxis = "US: Number of cases", label = "Infected forecast", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time, Recovered, yaxis = "US: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t_n, predictionn[3, :], xticks = ([0:15:30;], ["8 March' 20", "23 March' 20", "7 April' 20"]),  yaxis = "US: Number of cases", label = "Recovered forecast ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm,right_margin = 8mm,grid = :off, color = :blue, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing,background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)


savefig("US_4.pdf")

#New plots of uarantine population

S_NN_all_loss = predictionn[1, :]
I_NN_all_loss = predictionn[2, :]
R_NN_all_loss = predictionn[3, :]
T_NN_all_loss = predictionn[4, :]


 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

 for i = 1:length(S_NN_all_loss)
   Q_parameter[i] = abs(re(p3n[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1])
 end

scatter(t_n,Q_parameter/u0[1], ylabel = "Q(t)",xticks = ([0:15:30;], ["8 March' 20", "23 March' 20", "7 April' 20"]), label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, right_margin = 8mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("US_5.pdf")

#savefig("Nature_US_New_Final_Quarantine_Strength_Time.pdf")
using LaTeXStrings
scatter(t_n, abs(p3n[62]) ./ (abs(p3n[63]) .+ Q_parameter/u0[1]), xticks = ([0:15:30;], ["8 March' 20", "23 March' 20", "7 April' 20"]),  ylabel = "R(t)", label = "Effective reproduction number", right_margin = 8mm, legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = "R = 1")

savefig("US_6.pdf")
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

function QSIR_original(du, u, p, t)
    β = abs(0.27)
    γ = abs(-0.001322)
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - abs(re(p[1:61])(u)[1])*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  abs(re(p[1:61])(u)[1]*u[2]/u0[1])
end

function QSIR(du, u, p, t)
    β = abs(0.27)
    γ = abs(-0.001322)
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - ((33-t)/16)*abs(re(p[1:61])(u)[1])*u[2]/u0[1] - ((t-17)/16)*abs(re(p[64:124])(u)[1])*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  + ((33-t)/16)*abs(re(p[1:61])(u)[1])*u[2]/u0[1] + ((t-17)/16)*abs(re(p[64:124])(u)[1])*u[2]/u0[1]
end

u0 = Float64[327000000.0, 59131 ,285, 10]
tspan_n = (17.0, 33.0)
datasize_n = 16;
t_n = range(tspan_n[1],tspan_n[2],length=datasize_n)

D = load("US_QSIR.jld")
p3_US = D["Parameters"]
prob_n = ODEProblem(QSIR_original, u0, tspan_n, p3_US)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3_US,saveat=t_n))
scatter(Time, Infected,  yaxis = "US: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t_n, predictionn[2, :], yscale = :log, xticks = ([17:8:33;], ["25 March' 20", " 2 April' 20", "10 April' 20"]),  yaxis = "US: Number of cases", label = "US model", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :red, linewidth  = 3, ylims = (1, 20000000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

#Using US NN till Day 17 and then using Wuhan's NN
D = load("Mac_RD_Nature_QSIR.jld")
p3_Wuhan = D["Parameters"]
p3n = [p3_US; p3_Wuhan]

prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))
plot!(t_n, predictionn[2, :], xticks = ([17:8:33;], ["25 March' 20", " 2 April' 20", "10April' 20"]),  yaxis = "US: Number of cases", label = "Wuhan model", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (1, 1000000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

#Using US NN till Day 17 and then using Italy's NN
D = load("Italy_QSIR.jld")
p3_Italy = D["Parameters"]
p3n = [p3_US; p3_Italy]
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))
plot!(t_n, predictionn[2, :], xticks = ([17:8:33;], ["25 March' 20", " 2 April' 20", "10 April' 20"]),  yaxis = "US: Number of cases", label = "Italy model", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :green, linewidth  = 3, ylims = (1, 20000000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

#Using US NN till Day 17 and then using Korea's NN
D = load("Korea_QSIR.jld")
p3_Korea = D["Parameters"]
p3n = [p3_Korea; p3_Italy]
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))
plot!(t_n, predictionn[2, :], xticks = ([17:8: 33;], ["25 March' 20", " 2 April' 20", "10 April' 20"]),  yaxis = "US: Number of cases", label = " Korea model", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :purple, linewidth  = 3, ylims = (1, 20000000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("US_alln.pdf")
