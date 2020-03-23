using Plots
using Measures
using Flux
using DifferentialEquations
using DiffEqFlux

using MAT
vars = matread("C:/Users/Raj/Hubei_Track.mat")

Infected = vars["Hubei_Infected_All"]
Recovered = vars["Hubei_Recovered_All"]
Dead = vars["Hubei_Dead_All"]
Time = vars["Hubei_Time"]

ann = Chain(Dense(4,10,relu), Dense(10,1))
p1,re = Flux.destructure(ann)

function QSIR(du, u, p, t)
    β = p[62]
    γ = p[63]
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - re(p[1:61])(u)[1]*u[2]/u0[1]
    du[3] = γ*u[2]
    du[4] =  + re(p[1:61])(u)[1]*u[2]/u0[1]
end


u0 = Float64[11000000.0, 500 ,10, 10]

using JLD
D = load("Mac_RD_Nature_QSIR.jld")
S_NN_all_loss = D["S_NN_all_loss"]
I_NN_all_loss = D["I_NN_all_loss"]
R_NN_all_loss = D["R_NN_all_loss"]
T_NN_all_loss = D["T_NN_all_loss"]
t = D["t"]
Q_parameter = D["Q_parameter"]
p3 = D["Parameters"]

I = Infected[1, :]
R = Recovered[1,:]

scatter(Time[1:39], I[1:39], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, I_NN_all_loss, xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time[1:39], R[1:39], xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, R_NN_all_loss, xlims = (0, 50), xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 100000), foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

savefig("RD_Nature_QSIR_1_final.pdf")

scatter(t,Q_parameter/u0[1], xlims = (0, 50), xlabel = "Days since 24 Jan 2020", ylabel = "Q(t)", label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0.2, 0.8),  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_2_final.pdf")

#savefig("Nature_Wuhan_New_Final_Quarantine_Strength_Time.pdf")

scatter(t, p3[62] ./ (p3[63] .+ Q_parameter/u0[1]), xlims = (0, 50), xlabel = "Days since 24 Jan 2020", ylabel = "R(t)", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, ylims = (0.5, 2), yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) =1
plot!(f, color = :blue, linewidth = 3, label = "R = 1")

savefig("RD_Nature_QSIR_3_final.pdf")

scatter(t, T_NN_all_loss, xaxis = "Days since 24 Jan 2020", yaxis = "Wuhan: Number of cases", label = "Prediction: Quarantined", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :green, linewidth  = 3, ylims = (0, 1000000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_14final.pdf")

#Forecasting
p3n = p3
tspan_n = (0, 75.0)
datasize_n = 75;
t_n = range(tspan_n[1],tspan_n[2],length=datasize_n)
prob_n = ODEProblem(QSIR, u0, tspan_n, p3n)
predictionn = Array(concrete_solve(prob_n,Rosenbrock23(autodiff = false),u0,p3n,saveat=t_n))


scatter(Time[1:39], I[1:39], yaxis = "Wuhan: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t_n, predictionn[2, :], xticks = ([0:25:75;], ["24 Jan' 20", "18 Feb' 20", "14 March' 20", "8 April' 20"]),  yaxis = "Wuhan: Number of cases", label = "Infected forecast", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, color = :red, linewidth  = 3, ylims = (0, 80000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
#savefig("RD_Nature_QSIR_1.pdf")

scatter!(Time[1:39], R[1:39], yaxis = "Wuhan: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t_n, predictionn[3, :], xticks = ([0:25:75;], ["24 Jan' 20", "18 Feb' 20", "14 March' 20", "8 April' 20"]),  yaxis = "Wuhan: Number of cases", label = "Recovered forecast ", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, ylims = (0, 100000), foreground_color_legend = nothing, right_margin = 5mm, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)


savefig("RD_Nature_QSIR_10finaln.pdf")

#New plots of uarantine population

S_NN_all_loss = predictionn[1, :]
I_NN_all_loss = predictionn[2, :]
R_NN_all_loss = predictionn[3, :]
T_NN_all_loss = predictionn[4, :]


 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

 for i = 1:length(S_NN_all_loss)
   Q_parameter[i] = re(p3[1:61])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i], T_NN_all_loss[i]])[1]
 end

scatter(t_n,Q_parameter/u0[1], ylims = (0.4, 0.8), ylabel = "Q(t)", xticks = ([0:25:75;], ["24 Jan' 20", "18 Feb' 20", "14 March' 20", "8 April' 20"]), label = "Quarantine Strength",color = :black, framestyle = :box, grid =:off, legend = :bottomright, left_margin = 5mm, bottom_margin = 5mm, right_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
savefig("RD_Nature_QSIR_11final.pdf")

#savefig("Nature_Wuhan_New_Final_Quarantine_Strength_Time.pdf")
using LaTeXStrings
scatter(t_n, p3n[62] ./ (p3n[63] .+ Q_parameter/u0[1]), ylims = (0.5, 2), xticks = ([0:25:75;], ["24 Jan' 20", "18 Feb' 20", "14 March' 20", "8 April' 20"]),  ylabel = "R(t)", label = "Effective reproduction number", right_margin = 5mm, legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, color = :blue, linewidth = 3, label = "R = 1")

savefig("RD_Nature_QSIR_12final.pdf")
