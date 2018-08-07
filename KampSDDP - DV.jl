Pkg.add("Clp")
Pkg.add("StochDynamicProgramming")
using JuMP, Clp, StochDynamicProgramming

# Define hydrostation
mutable struct Hydrostation
  ID::Int64
  Name::String
  a::Float64
  b::Float64
  H_ref::Float64
  V_min::Float64
  V_max::Float64
  C_min::Float64
  C_max::Float64
  S_min::Float64
  S_max::Float64
  Stocks::Vector{Float64}
  Controls::Vector{Float64}
end

# Set the timeframe depending on the size of the dataframe
const timeframe = size(df_Strompreis)[1]

# test
hourlyValue = 60*60/1000

# Set values for Ottenstein
OTT = Hydrostation(1, # ID
        "Ottenstein", # Name
        15.8686281827966, # a
        2.24102766240712, # b
        452,              # H_ref
        15.8686281827966*(df_Ott_Low_Limit[:Wert][1]-452)^2.24102766240712, # V_min
        15.8686281827966*(df_Ott_Upp_Limit[:Wert][1]-452)^2.24102766240712, # V_max
        0*hourlyValue,      # C_min
        97.41*hourlyValue,  # C_max
        0*hourlyValue,  # S_min
        0*hourlyValue,  # S_max
        zeros(timeframe), # Stocks
        zeros(timeframe)  # Controls
      )
#

# Set values for Dobra-Krumau
DOB = Hydrostation(2, # ID
        "Dobra-Krumau", # Name
        16.4711556389479, # a
        2.07566579351536, # b
        405,              # H_ref
        16.4711556389479*(df_Dob_Low_Limit[:Wert][1]-405)^2.07566579351536, # V_min
        16.4711556389479*(df_Dob_Upp_Limit[:Wert][1]-405)^2.07566579351536, # V_max
        0*hourlyValue,      # C_min
        29.27*hourlyValue,  # C_max
        0*hourlyValue,  # S_min
        0*hourlyValue,  # S_max
        zeros(timeframe), # Stocks
        zeros(timeframe)  # Controls
      )
#

# Set values for Thurnberg-Wegscheid
THU = Hydrostation(3, # ID
        "Thurnberg-Wegscheid", # Name
        56.2543470117431, # a
        1.65830720723156, # b
        354,              # H_ref
        56.2543470117431*(df_Thu_Low_Limit[:Wert][1]-354)^1.65830720723156, # V_min
        56.2543470117431*(df_Thu_Upp_Limit[:Wert][1]-354)^1.65830720723156, # V_max
        3*hourlyValue,     # C_min
        16.2*hourlyValue,  # C_max
        0*hourlyValue,  # S_min
        0*hourlyValue,  # S_max
        zeros(timeframe), # Stocks
        zeros(timeframe)  # Controls
      )
#

# Set the solver
solver = ClpSolver()

# Set tolerance at 0.1% for convergence
epsilon = 0.001

# Set the cost vector
cost = -df_Strompreis[:Wert]

# Define the dynamics
function dynamic(t, x, u, w)
    return [x[1] - u[1] - u[4] + w[1], x[2] - u[2] - u[5] + u[1] + u[4] + w[2], x[3] - u[3] - u[6] + u[2] + u[5] + w[3]]
end

# Define the cost function
function cost_t(t, x, u, w)
    return cost[t] * (u[1] + u[2] + u[3])
end

# Create a law of noises
# NoiseLaw vector in SDDP
function generate_probability_laws(n_scenarios)
    aleas1 = Array{Float64}(transpose(df_Ott_Inflow[:Wert])*hourlyValue)
    aleas2 = Array{Float64}(transpose(df_Dob_Inflow[:Wert])*hourlyValue)
    aleas3 = Array{Float64}(transpose(df_Thu_Inflow[:Wert])*hourlyValue)
    aleas = vcat(aleas1, aleas2, aleas3)

    laws = Vector{NoiseLaw}(timeframe)
    proba = 1/n_scenarios*ones(n_scenarios)

    for t=1:timeframe
        laws[t] = NoiseLaw(aleas[:, [t, t, t]], proba)
    end

    return laws
end

aleas = generate_probability_laws(1)

# Set bounds on the state
x_bounds = [(OTT.V_min, OTT.V_max), (DOB.V_min, DOB.V_max), (THU.V_min, THU.V_max)]

# Set bounds on the control
u_bounds = [(OTT.C_min, OTT.C_max), (DOB.C_min, DOB.C_max), (THU.C_min, THU.C_max), (OTT.S_min, OTT.S_max), (DOB.S_min, DOB.S_max), (THU.S_min, THU.S_max)]

# Set the start values
x0 = [median([OTT.V_min, OTT.V_max]), median([DOB.V_min, DOB.V_max]), median([THU.V_min, THU.V_max])]

# Create the model
model = LinearSPModel(timeframe, # number of timestep
                    u_bounds, # control bounds
                    x0, # initial state
                    cost_t, # cost function
                    dynamic, # dynamic function
                    aleas);

# Add the bounds to the model
set_state_bounds(model, x_bounds)

# Set the SDDP parameters
params = SDDPparameters(solver,
                        passnumber=3,
                        gap=epsilon,
                        montecarlo_in_iter=3,
                        max_iterations=3)

# Solve the SDDP
sddp = @time solve_SDDP(model, params, 1)
# ~50 sec (für 30 T)
# ~593 sec (für 365 T)

# Start the simulation
costs, stocks, controls = StochDynamicProgramming.simulate(sddp, 1)

# Get the SDDP cost
SDDP_COST = costs[1]

# Get the stock results for a hydrostation
function GetStocks(hydrostation::Hydrostation)
  id = hydrostation.ID
  stocksTemp = hydrostation.Stocks

  # Convert volume to water level
  function VolumeToWaterlevel(volume::Float64, hydrostation::Hydrostation)
    # (V/a)^(1/b)+H_ref
    return (volume/hydrostation.a)^(1/hydrostation.b)+hydrostation.H_ref
  end

  for i in eachindex(stocks[:,1,id])
    volume = stocks[i,1,id]
    stocksTemp[i] = VolumeToWaterlevel(volume, hydrostation)
  end

  return stocksTemp
end

OTT.Stocks = GetStocks(OTT)
DOB.Stocks = GetStocks(DOB)
THU.Stocks = GetStocks(THU)
OTT.Controls = controls[:, 1, OTT.ID]/hourlyValue
DOB.Controls = controls[:, 1, DOB.ID]/hourlyValue
THU.Controls = controls[:, 1, THU.ID]/hourlyValue

#pumpControls = controls[:, 1, 7]/hourlyValue
