#= using StochDynamicProgramming, Clp
println("library loaded")

run_sddp = true # false if you don't want to run sddp
run_sdp  = true # false if you don't want to run sdp
test_simulation = true # false if you don't want to test your strategies

######## Optimization parameters  ########
# choose the LP solver used.
const SOLVER = ClpSolver() 			   # require "using Clp"
#const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0) # require "using CPLEX"

# convergence test
const MAX_ITER = 10 # number of iterations of SDDP

const step = 0.1   # discretization step of SDP

######## Stochastic Model  Parameters  ########
const N_STAGES = 6              # number of stages of the SP problem
const N_STOCKS = 3              # number of stocks of the SP problem
const COSTS = [sin(3*t)-1 for t in 1:N_STAGES]
#const COSTS = rand(N_STAGES)    # randomly generating deterministic costs


const CONTROL_MAX = 0.5         # bounds on the control
const CONTROL_MIN = 0

const XI_MAX = 0.3              # bounds on the noise
const XI_MIN = 0
const N_XI = 3                 # discretization of the noise

const r = 0.5                  # bound on cumulative control : \sum_{i=1}^N u_i < rN

const S0 = [0.5 for i=1:N_STOCKS]     # initial stock

# create law of noises
proba = 1/N_XI*ones(N_XI) # uniform probabilities
xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
xi_law = StochDynamicProgramming.noiselaw_product([NoiseLaw(xi_support, proba) for i=1:N_STOCKS]...)
xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]

# Define dynamic of the stock:
function dynamic(t, x, u, xi)
    return [x[i] + u[i] - xi[i] for i in 1:N_STOCKS]
end


# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COSTS[t] *sum(u)
end

# constraint function
constraints_dp(t, x, u, w) = sum(u) <= r*N_STOCKS
constraints_sddp(t, x, u, w) = [sum(u) - r*N_STOCKS]

######## Setting up the SPmodel
s_bounds = [(0, 1) for i = 1:N_STOCKS]			# bounds on the state
u_bounds = [(CONTROL_MIN, CONTROL_MAX) for i = 1:N_STOCKS] # bounds on controls
spmodel = LinearSPModel(N_STAGES,u_bounds,S0,cost_t,dynamic,xi_laws, ineqconstr=constraints_sddp)
set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
println("Model set up")

######### Solving the problem via SDDP
if run_sddp
    tic()
    println("Starting resolution by SDDP")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    V, pbs = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, V)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

######### Solving the problem via Dynamic Programming
if run_sdp
    tic()
    println("Starting resolution by SDP")
    stateSteps = [step for i=1:N_STOCKS] # discretization step of the state
    controlSteps = [step for i=1:N_STOCKS] # discretization step of the control
    infoStruct = "HD" # noise at time t is known before taking the decision at time t

    paramSDP = SDPparameters(spmodel, stateSteps, controlSteps, infoStruct)
    spmodel_sdp = StochDynamicProgramming.build_sdpmodel_from_spmodel(spmodel)
    spmodel_sdp.constraints = constraints_dp

    Vs = solve_DP(spmodel_sdp, paramSDP, 1)
    value_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
    println("Value obtained by SDP: "*string(round(value_sdp,4)))
    toc(); println();
end

######### Comparing the solutions on simulated scenarios.
#srand(1234) # to fix the random seed accross runs
if run_sddp && run_sdp && test_simulation
    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)
    costsddp, stocks = forward_simulations(spmodel, paramSDDP, pbs, scenarios)
    costsdp, states, controls = sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)
    println("Simulated relative gain of sddp over sdp: "
            *string(round(200*mean(costsdp-costsddp)/abs(mean(costsddp+costsdp)),3))*"%")
end



Pkg.clone("https://github.com/odow/SDDP.jl.git")
using SDDP, JuMP, Clp, Base.Test

srand(100)

XI = collect(Base.product([linspace(0, 0.3, 3) for i in 1:3]...))[:]

m = SDDPModel(
                  sense = :Min,
                 stages = 5,
                 solver = ClpSolver(),
        objective_bound = -5
                                ) do sp, stage

    @state(sp, 0 <= stock[i=1:3] <= 1, stock0 == 0.5)

    @variable(sp, 0 <= control[i=1:3] <= 0.5)

    @constraint(sp, sum(control) - 0.5 * 3 <= 0)

    @rhsnoises(sp, xi = XI, begin
        stock[1] == stock0[1] + control[1] - xi[1]
        stock[2] == stock0[2] + control[2] - xi[2]
        stock[3] == stock0[3] + control[3] - xi[3]
    end)

    @stageobjective(sp, (sin(3 * stage) - 1) * sum(control))
end

@time status = SDDP.solve(m, max_iterations = 100,
print_level = 0)
@test isapprox(SDDP.getbound(m), -4.349, atol=0.01)

results = simulate(m, 5000)
@test length(results) == 5000
@test isapprox(mean(r[:objective] for r in results), -4.349, atol=0.02)



using JuMP, SDDP, Clp, Base.Test

struct PriceTurbine
    flowknots::Vector{Float64}
    powerknots::Vector{Float64}
end

struct PriceReservoir
    min::Float64
    max::Float64
    initial::Float64
    turbine::PriceTurbine
    spill_cost::Float64
    inflows::Vector{Float64}
end

function priceprocess(USE_AR1)
    b_t = [61.261, 56.716, 59.159, 66.080, 72.131, 76.708, 76.665, 76.071, 76.832, 69.970, 69.132, 67.176]
    alpha = 0.5
    beta = -0.5
    minprice = 40.0
    maxprice = 100.0
    pbar = 61.261
    noise = [0.5,1.5,2.5,3.5,4.5]
    NOISES = DiscreteDistribution(vcat(-reverse(noise), noise))

    function ar1price(price, noise, stage, markovstate)
        if stage > 1
            return alpha * price + (1-alpha) * b_t[stage] + noise
        else
            return price
        end
    end

    function ar2price(price, noise, stage, markovstate)
        # price is a Tuple{Float64, Float64}
        # price[1] is t-1, price[2] is t-2
        if stage > 1
            return (alpha * price[1] + (1-alpha) * b_t[stage] + beta * (price[1] - price[2]) + noise, price[1])
        else
            return price
        end
    end

    if USE_AR1
        return DynamicPriceInterpolation(
            dynamics       = ar1price,
            initial_price  = pbar,
            min_price      = minprice,
            max_price      = maxprice,
            noise          = NOISES
        )
    else
        return DynamicPriceInterpolation(
            dynamics       = ar2price,
            initial_price  = (pbar, pbar),
            min_price      = (minprice, minprice),
            max_price      = (maxprice, maxprice),
            noise          = NOISES
        )
    end
end

function buildmodel(USE_AR1, valley_chain)
    valuefunction = priceprocess(USE_AR1)
    return SDDPModel(
                sense           = :Min,
                stages          = 12,
                objective_bound = 50_000.0,
                solver          = ClpSolver(),
                value_function   = valuefunction
                                        ) do sp, stage
        N = length(valley_chain)
        turbine(i) = valley_chain[i].turbine
        @state(sp, valley_chain[r].min <= reservoir[r=1:N] <= valley_chain[r].max, reservoir0==valley_chain[r].initial)
        @variables(sp, begin
            70 >= outflow[r=1:N]      >= 0
            spill[r=1:N]        >= 0
            generation_quantity >= 0 # Total quantity of water
            # Proportion of levels to dispatch on
            0 <= dispatch[r=1:N, level=1:length(turbine(r).flowknots)] <= 1
        end)
        @constraints(sp, begin
            # flow from upper reservoir
            reservoir[1] == reservoir0[1] - outflow[1] - spill[1]
            # other flows
            flow[i=2:N], reservoir[i] == reservoir0[i] - outflow[i] - spill[i] + outflow[i-1] + spill[i-1]
            # Total quantity generated
            generation_quantity == sum(turbine(r).powerknots[level] * dispatch[r,level] for r in 1:N for level in 1:length(turbine(r).powerknots))
            # Flow out
            turbineflow[r=1:N], outflow[r] == sum(turbine(r).flowknots[level] * dispatch[r, level] for level in 1:length(turbine(r).flowknots))
            # Dispatch combination of levels
            dispatched[r=1:N], sum(dispatch[r, level] for level in 1:length(turbine(r).flowknots)) <= 1

            maxflow[i=1:N], outflow[i] <= reservoir0[i]
        end)
        if USE_AR1
            @stageobjective(sp,
                price -> sum(
                    valley_chain[i].spill_cost * spill[i] for i in 1:N
                    ) - price * generation_quantity
            )
        else
            @stageobjective(sp,
                price -> sum(
                    valley_chain[i].spill_cost * spill[i] for i in 1:N
                    ) - price[1] * generation_quantity
            )
        end
    end
end

function runpaper(USE_AR1, valley_chain, name)
    m = buildmodel(USE_AR1, valley_chain)
    solve(m, max_iterations = 2_000, log_file="$(name).log")

    SIMN = 1_000
    sim = simulate(m, SIMN, [:reservoir, :price, :generation_quantity])
    plt = SDDP.newplot()
    if USE_AR1
        SDDP.addplot!(plt, 1:SIMN, 1:12, (i, t)->sim[i][:price][t], title="Simulated Price", ylabel="Price (\$/MWH)")
    else
        SDDP.addplot!(plt, 1:SIMN, 1:12, (i, t)->sim[i][:price][t][1], title="Simulated Price", ylabel="Price (\$/MWH)")
    end
    SDDP.addplot!(plt, 1:SIMN, 1:12, (i, t)->sim[i][:generation_quantity][t], title="Offer", ylabel="Quantity (MWH)")
    for r in 1:length(valley_chain)
        SDDP.addplot!(plt, 1:SIMN, 1:12, (i, t)->sim[i][:reservoir][t][r], title="Storage (Reservoir $(r))", ylabel="Volume (m^3)")
    end
    SDDP.show("$(name).html", plt)
end



 runpaper(
    true,
     [
         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0])
     ],
     "example_one"
 )

# runpaper(
#     false,
#     [
#         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
#         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
#         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
#         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
#         PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0])
#     ],
#     "example_two"
# )

srand(123)
m = buildmodel(false, [
        PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
        PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0])
    ])
srand(123)
solve(m, max_iterations = 20, print_level=0, cut_output_file="river.cuts")
@test getbound(m) >= -40_000

# test cut round trip with multiple price dimensions
srand(123)
m2 = buildmodel(false, [
        PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0]),
        PriceReservoir(0, 200, 100, PriceTurbine([50, 60, 70], [55, 65, 70]), 1000, [0])
    ])
try
    SDDP.loadcuts!(m2, "river.cuts")
finally
    rm("river.cuts")
end
srand(123)
SDDP.solve(m, max_iterations=1, print_level=0)
srand(123)
SDDP.solve(m2, max_iterations=1, print_level=0)
@test isapprox(getbound(m2), getbound(m), atol=1e-4)





import StochDynamicProgramming, Distributions
println("library loaded")

# We have to define the instance on all the workers (processes)
@everywhere begin

    run_sdp = true

    ######## Stochastic Model  Parameters  ########
    const N_STAGES = 50
    const COSTS = rand(N_STAGES)
    const DEMAND = rand(N_STAGES)

    const CONTROL_MAX = 0.5
    const CONTROL_MIN = 0

    const STATE_MAX = 1
    const STATE_MIN = 0

    const XI_MAX = 0.3
    const XI_MIN = 0
    const N_XI = 10
    # initial stock
    const S0 = 0.5

    # charge and discharge efficiency parameters
    const rho_c = 0.98
    const rho_dc = 0.97

    # create law of noises
    proba = 1/N_XI*ones(N_XI) # uniform probabilities
    xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
    xi_law = StochDynamicProgramming.NoiseLaw(xi_support, proba)
    xi_laws = StochDynamicProgramming.NoiseLaw[xi_law for t in 1:N_STAGES-1]

    # Define dynamic of the stock:
    function dynamic(t, x, u, xi)
    	return [ x[1] + 1/rho_dc * min(u[1],0) + rho_c * max(u[1],0) ]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, xi)
        return COSTS[t] * max(0, DEMAND[t] + u[1] - xi[1])
    end

    function constraint(t, x, u, xi)
    	return true
    end

    function finalCostFunction(x)
    	return(0)
    end

    ######## Setting up the SPmodel
    s_bounds = [(STATE_MIN, STATE_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]
    spmodel = StochDynamicProgramming.StochDynProgModel(N_STAGES, s_bounds,
                                                                    u_bounds,
                                                                    [S0],
                                                                    cost_t,
                                                                    finalCostFunction,
                                                                    dynamic,
                                                                    constraint,
                                                                    xi_laws)

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

    stateSteps = [0.01]
    controlSteps = [0.001]
    infoStruct = "HD" # noise at time t is not known before taking the decision at time t

    paramSDP = StochDynamicProgramming.SDPparameters(spmodel, stateSteps,
                                                    controlSteps, infoStruct)
end

Vs = StochDynamicProgramming.solve_dp(spmodel,paramSDP, 1)

lb_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
println("Value obtained by SDP: "*string(lb_sdp))
costsdp, states, stocks = StochDynamicProgramming.forward_simulations(spmodel,paramSDP,Vs,scenarios)
println(mean(costsdp))
=#






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
