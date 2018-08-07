# Load packages for the SDDP
using JuMP, Clp, StochDynamicProgramming, Interpolations

# Define the pump
mutable struct Pump
  ID::Int64
  C_min::Float64
  C_max::Float64
  ElectricalPower::Vector{Float64}
  Waterflow::Vector{Float64}
  Controls::Array{Float64,2}
end

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
  ElectricalPower::Vector{Float64}
  Waterflow::Vector{Float64}
  Stocks::Array{Float64,2}
  Controls::Array{Float64,2}
  Spills::Array{Float64,2}
  ShadowPrices::Array{Float64}
  Pump
end

# Convert String to DateTime
function To_DateTime(datetime::String)
  tmp = split(datetime, ['-', ' '])
  return DateTime(parse(tmp[1]),parse(tmp[2]),parse(tmp[3]),parse(tmp[4]))
end

# Convert Volume to Water-level
function To_Waterlevel(volume, hydrostation::Hydrostation)
  # (V/a)^(1/b)+H_ref
  return (volume./hydrostation.a).^(1./hydrostation.b).+hydrostation.H_ref
end

# Convert Water-level to Volume
function To_Volume(waterlevel, hydrostation::Hydrostation)
  # a*(p-H_ref)^b
  return hydrostation.a*(waterlevel-hydrostation.H_ref).^hydrostation.b
end

# Maps between power and flow rate
function interp1(x, machine::Any, toMW::Bool=false)
  y = zeros(x)
  idx = trues(x)
  wf, ep = machine.Waterflow, machine.ElectricalPower

  intf = toMW ? interpolate((wf,), ep, Gridded(Linear())) : interpolate((ep,), wf, Gridded(Linear()))

  y[idx] = intf[x[idx]]
  return y
end

# Convert m³/s to Tm³/h
function To_Tm³𝞘h(value)
  return value*60*60/1000
end

# Convert Tm³/h to m³/s
function To_m³𝞘s(value)
  return value/(60*60/1000)
end

# Set the timeFrame depending on the size of the dataframe
const timeFrame = size(df_Strompreis)[1]

# Set values for Ottenstein
OTT = Hydrostation(1,     # ID
        "Ottenstein",     # Name
        15.8686281827966, # a
        2.24102766240712, # b
        452,              # H_ref
        15.8686281827966*(df_Ott_Low_Limit[:Wert][1]-452).^2.24102766240712, # V_min
        15.8686281827966*(df_Ott_Upp_Limit[:Wert][1]-452).^2.24102766240712, # V_max
        To_Tm³𝞘h(0),      # C_min
        To_Tm³𝞘h(97.41),  # C_max
        To_Tm³𝞘h(0),      # S_min
        To_Tm³𝞘h(100),    # S_max
        [0.0, 0.01, 12.0, 24.0, 36.0, 48.0],  # ElectricalPower
        [0.0, 1.4, 24.1, 48.5, 72.9, 97.4],   # Waterflow
        zeros(timeFrame,2), # Stocks
        zeros(timeFrame,2), # Controls
        zeros(timeFrame,2), # Spills
        zeros(timeFrame),   # Shadow Prices
        Pump(7,           # ID
            To_Tm³𝞘h(0),        # C_min
            To_Tm³𝞘h(27.31),    # C_max
            [0.0, 9.12, 18.25],   # ElectricalPower
            [0.0, 13.65, 27.31],  # Waterflow
            zeros(timeFrame,2)))    # Controls
#

# Set values for Dobra-Krumau
DOB = Hydrostation(2,     # ID
        "Dobra-Krumau",   # Name
        16.4711556389479, # a
        2.07566579351536, # b
        405,              # H_ref
        16.4711556389479*(df_Dob_Low_Limit[:Wert][1]-405).^2.07566579351536, # V_min
        16.4711556389479*(df_Dob_Upp_Limit[:Wert][1]-405).^2.07566579351536, # V_max
        To_Tm³𝞘h(0),      # C_min
        To_Tm³𝞘h(29.27),  # C_max
        To_Tm³𝞘h(0),      # S_min
        To_Tm³𝞘h(0),      # S_max
        [0.0, 0.5, 5.7, 11.4, 17.1],  # ElectricalPower
        [0.0, 1.2, 9.5, 19.1, 29.3],  # Waterflow
        zeros(timeFrame,2), # Stocks
        zeros(timeFrame,2), # Controls
        zeros(timeFrame,2), # Spills
        zeros(timeFrame),   # Shadow Prices
        nothing)
#

# Set values for Thurnberg-Wegscheid
THU = Hydrostation(3,     # ID
        "Thurnberg-Wegscheid", # Name
        56.2543470117431, # a
        1.65830720723156, # b
        354,              # H_ref
        56.2543470117431*(df_Thu_Low_Limit[:Wert][1]-354).^1.65830720723156, # V_min
        56.2543470117431*(df_Thu_Upp_Limit[:Wert][1]-354).^1.65830720723156, # V_max
        To_Tm³𝞘h(3),      # C_min
        To_Tm³𝞘h(16.2),   # C_max
        To_Tm³𝞘h(0),      # S_min
        To_Tm³𝞘h(0),      # S_max
        [0.0, 0.3, 1.4, 2.7],   # ElectricalPower
        [0.0, 2.0, 8.1, 16.2],  # Waterflow
        zeros(timeFrame,2), # Stocks
        zeros(timeFrame,2), # Controls
        zeros(timeFrame,2), # Spills
        zeros(timeFrame),   # Shadow Prices
        nothing)
#

# Creates the SDDP model
function CreateSDDPModel(fwrd_passes::Int64, max_iter::Int64, monte_carlo::Int64, endRestr::Bool=false)
  # Set the solver
  solver = ClpSolver()

  # Set tolerance at 5% for convergence
  epsilon = 0.05

  # Set the cost vector
  cost = - df_Strompreis[:Wert]

  # Define the dynamics
  function dynamic(t, x, u, w)
           #          Ottenstein            #                     Dobra                     #                 Thurnberg              #
    return [x[1] - u[1] - u[4] + w[1] + u[7], x[2] - u[2] - u[5] + u[1] + u[4] + w[2] - u[7], x[3] - u[3] - u[6] + u[2] + u[5] + w[3]]
  end

  # Define the cost function
  function cost_t(t, x, u, w)
    return cost[t] * (u[1]/2 + u[2]/1.66 + u[3]/6 - u[7]/1.5) / 3.6
  end

  # Create a law of noises
  function generate_probability_laws(n_scenarios)
    OTT_prob = zeros(timeFrame, n_scenarios)
    d = 24

    for t in 1:d:timeFrame
      val = mean(To_Tm³𝞘h(df_Ott_Inflow[:Wert])[t:t+d-1])

      for i in 1:n_scenarios
        OTT_prob[t:t+d-1,i] = val*(1.3-(1/i))
      end
    end

    laws = Vector{NoiseLaw}(timeFrame)
    proba = 1/n_scenarios*ones(n_scenarios)

    for t = 1:timeFrame
      OTT_aleas = OTT_prob[t,:]'
      DOB_aleas = repeat([To_Tm³𝞘h(df_Dob_Inflow[:Wert][t])], inner=[n_scenarios])'
      THU_aleas = repeat([To_Tm³𝞘h(df_Thu_Inflow[:Wert][t])], inner=[n_scenarios])'
      aleas = vcat(OTT_aleas, DOB_aleas, THU_aleas)
      laws[t] = NoiseLaw(aleas, proba)
    end

    return laws
  end

  aleas = generate_probability_laws(5)

  # Set bounds on the state
  x_bounds = [(To_Volume(df_Ott_Low_Limit[:Wert][1], OTT), To_Volume(df_Ott_Upp_Limit[:Wert][1], OTT)),
              (To_Volume(df_Dob_Low_Limit[:Wert][1], DOB), To_Volume(df_Dob_Upp_Limit[:Wert][1], DOB)),
              (To_Volume(df_Thu_Low_Limit[:Wert][1], THU), To_Volume(df_Thu_Upp_Limit[:Wert][1], THU))]

  # Set bounds on the control
  u_bounds = [(OTT.C_min, OTT.C_max), (DOB.C_min, DOB.C_max), (THU.C_min, THU.C_max), (OTT.S_min, OTT.S_max), (DOB.S_min, DOB.S_max), (THU.S_min, THU.S_max), (OTT.Pump.C_min, OTT.Pump.C_max)]

  # Set the start values
  x0 = [median([OTT.V_min[1], OTT.V_max[1]]), median([DOB.V_min[1], DOB.V_max[1]]), median([THU.V_min[1], THU.V_max[1]])]

  # Penalize the final costs if it is greater than the initial value
  function final_cost(model, m)
    alpha = m[:alpha]
    w = m[:w]
    x = m[:x]
    u = m[:u]
    xf = m[:xf]
    OTT_end = x0[OTT.ID]
    DOB_end = x0[DOB.ID]
    THU_end = x0[THU.ID]
    @JuMP.variable(m, z1 >= 0.)
    @JuMP.variable(m, z2 >= 0.)
    @JuMP.variable(m, z3 >= 0.)
    @JuMP.constraint(m, alpha == 0.)
    @JuMP.constraint(m, z1 >= OTT_end - xf[1])
    @JuMP.constraint(m, z2 >= DOB_end - xf[2])
    @JuMP.constraint(m, z3 >= THU_end - xf[3])
    @JuMP.objective(m, Min, model.costFunctions(model.stageNumber-1, x, u, w) + 500*(z1+z2+z3))
  end

  final_restriction = endRestr ? final_cost : nothing

  # Create the model
  model = LinearSPModel(timeFrame, # number of timestep
                      u_bounds, # control bounds
                      x0, # initial state
                      cost_t, # cost function
                      dynamic, # dynamic function
                      Vfinal=final_restriction,
                      aleas)

  # Add the bounds to the model
  set_state_bounds(model, x_bounds)

  # Set the SDDP parameters
  params = SDDPparameters(solver,
                          passnumber=fwrd_passes,
                          gap=epsilon,
                          montecarlo_final=monte_carlo,
                          montecarlo_in_iter=monte_carlo,
                          max_iterations=max_iter)

  return model, params
end

function SDDP(fwrd_passes::Int64, max_iter::Int64, monte_carlo::Int64, simulations::Int64, endRestr::Bool=false)
  # Create model
  model, params = CreateSDDPModel(fwrd_passes, max_iter, monte_carlo, endRestr)
  # Solve SDDP
  sddp = solve_SDDP(model, params, 2)
  # Get results
  costs, stocks, controls = StochDynamicProgramming.simulate(sddp, simulations)

  # Set the results to the model
  for hs in hydrostations
    # Set stocks
    hs.Stocks = To_Waterlevel(stocks[:,:,hs.ID], hs)
    # Set controls
    hs.Controls = To_m³𝞘s(interp1(controls[:, :, hs.ID], hs, true))
    # Set pump controls (if available)
    hs.Pump.Controls != nothing ? To_m³𝞘s(interp1(controls[:, :, hs.Pump.ID], hs.Pump, true)) : nothing
    # Set spills
    hs.Spills = To_m³𝞘s(controls[:, :, hs.ID+3])

    # Get the shadow prices
    for i in eachindex(sddp.solverinterface)
      x = StochDynamicProgramming._getdual(sddp.solverinterface[i])
      hs.ShadowPrices[i] = x[7-hs.ID]
    end
  end
  # return costs
  return mean(abs(costs))
end

# Benchmarks SDDP with different input params
function Benchmark_SDDP(fwrd_passes::Vector{Int64}, max_iter::Vector{Int64}, monte_carlo::Vector{Int64}=[0])
  results = ([], [], [])

  for mo in monte_carlo
    for it in max_iter
      for fw in fwrd_passes
        model, params = CreateSDDPModel(fw, it, mo)
        sddp = solve_SDDP(model, params, 2)
        execTime = sum(sddp.stats.exectime)
        costs = mean(StochDynamicProgramming.simulate(sddp, 10)[1])
        println("Execution time: $execTime")
        push!(results[1], execTime)
        println("Costs: $costs")
        push!(results[2], costs)
        push!(results[3], sddp)
      end
    end
  end
  # returns exection times, simulated costs, sddp
  return results
end

results = Benchmark_SDDP([10,20],[15])
