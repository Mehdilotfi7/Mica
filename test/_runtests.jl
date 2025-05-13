using Test
using TSCPDetector.ModelSimulation  # Assuming TSCPDetector is your package

# =============================================================================
# Test DataHandler Module
# =============================================================================

using Test
using DataFrames
using YourPackageName.DataHandler  # Replace with your actual module path

@testset "DataHandler Module Tests" begin

    @testset "DataFrame with Missing Values" begin
        df = DataFrame(x = [1, 2, 3, 4, 5, 6],
                       y = [10, 20, 30, 40, 50, 60])
        mat, info = preprocess_data(df)
        @test size(mat) == (6, 2)
        @test !info.is_data_vector
    end

    @testset "Matrix Input" begin
        mat_input = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        mat, info = preprocess_data(mat_input)
        @test mat == mat_input
        @test !info.is_data_vector
    end

    @testset "1D Vector Input" begin
        vec = [1.0, 2.0, 3.0, 4.0]
        mat, info = preprocess_data(vec)
        @test mat == reshape(vec, :, 1)
        @test info.is_data_vector
    end

    @testset "Vector of Vectors Input" begin
        vvec = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mat, info = preprocess_data(vvec)
        @test size(mat) == (2, 3)
    end

    @testset "DataFrame without Missing Values" begin
        df_nomissing = DataFrame(a = 1:3, b = 4:6)
        mat, info = preprocess_data(df_nomissing)
        @test mat == Matrix(df_nomissing)
        @test !info.is_data_vector
    end

end


# =============================================================================
# Test ModelSimulation Module
# =============================================================================

@testset "ModelSimulation Tests" begin

    # ----------------------
    # Test ODE Model
    # ----------------------
    @testset "ODE Model Test" begin
        params = Dict(:p => [0.5])
        initial_conditions = [1.0]
        tspan = (0.0, 5.0)
        
        model = ODEModelSpec(exponential_ode_model, params, initial_conditions, tspan)
        result = simulate_model(model)

        @test isa(result, DataFrame)
        @test size(result, 1) == 100  # Should have 100 time points
        @test all(result.state .>= 0.0)  # Exponential decay, should be non-negative
    end

    # ----------------------
    # Test Difference Model
    # ----------------------
    @testset "Difference Model Test" begin
        params = Dict(:θ1 => 0.1, :θ2 => 0.2, :θ3 => 0.3, :θ4 => 0.4, :θ5 => 0.5, :θ6 => 0.6, :θ7 => 1.0)
        initial_condition = 10.0
        num_steps = 50
        wind_speeds = rand(num_steps)
        ambient_temperatures = rand(num_steps)

        model = DifferenceModelSpec(example_difference_model, params, initial_condition, num_steps, (wind_speeds, ambient_temperatures))
        result = simulate_model(model)

        @test isa(result, DataFrame)
        @test size(result, 1) == num_steps
    end

    # ----------------------
    # Test Regression Model
    # ----------------------
    @testset "Regression Model Test" begin
        params = Dict(:a => 2.0, :b => 1.0)
        time_steps = 30

        model = RegressionModelSpec(example_regression_model, params, time_steps)
        result = simulate_model(model)

        @test isa(result, DataFrame)
        @test size(result, 1) == time_steps
        @test all(result.simulated_values .== params[:a] .* result.time .+ params[:b])
    end

end

# =============================================================================
# Test Model Handling Module
# =============================================================================

@testset "Model Handling + Objective Function Tests" begin

    # =========================================================================
    # Test ODE Model
    # =========================================================================
    @testset "ODE Model" begin
        ode_params = Dict(:p => [0.5])
        ode_u0 = [1.0]
        tspan = (0.0, 10.0)

        base_model = ODEModelSpec(_ModelSimulation.exponential_ode_model, ode_params, ode_u0, tspan)
        manager = ModelManager(base_model)

        # Check initial condition accessor
        @test get_initial_condition(manager) == ode_u0
        @test get_model_type(manager) == "ODE"

        # segment_model
        seg_pars = [0.05] # update for ODE
        parnames = [:p]
        new_ode_model = segment_model(manager, seg_pars, parnames, 0, 2, [1.0])
        @test new_ode_model.params[:p] == 0.05
        @test new_ode_model.tspan == (0, 2.0)

        # Update initial condition
        sim = simulate_model(base_model)
        @test isa(sim, DataFrame)
        @test update_initial_condition(manager, sim) == [sim.state[end]]
    end

    # =========================================================================
    # Test Difference Model
    # =========================================================================
    @testset "Difference Model" begin
        num_steps = 50
        wind = rand(num_steps)
        temp = rand(num_steps)
        pars = Dict(:θ1 => 0.1, :θ2 => 0.2, :θ3 => 0.3, :θ4 => 0.4, :θ5 => 0.5, :θ6 => 0.6, :θ7 => 1.0)
        ic = 10.0

        diff_model = DifferenceModelSpec(_ModelSimulation.example_difference_model, pars, ic, num_steps, (wind, temp))
        diff_manager = ModelManager(diff_model)

        # Check initial condition accessor
        @test get_model_type(diff_manager) == "Difference"
        @test get_initial_condition(diff_manager) == ic

        # segment_model
        seg_pars = fill(0.2, 7) # update for Difference
        parnames = [:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7]
        new_diff_model = segment_model(diff_manager, seg_pars, parnames, 10, 20, 0.0)
        @test new_diff_model.num_steps == 11
        @test new_diff_model.extra_data[1] == wind[10:20]

        # Update initial condition
        diff_sim = simulate_model(diff_model)
        @test update_initial_condition(diff_manager, diff_sim) == diff_sim.state[end]
    end

    # =========================================================================
    # Test Regression Model
    # =========================================================================
    @testset "Regression Model" begin
        pars = Dict(:a => 2.0, :b => 1.0)
        time_steps = 30
        reg_model = RegressionModelSpec(_ModelSimulation.example_regression_model, pars, time_steps)
        reg_manager = ModelManager(reg_model)

        #Check initial condition accessor
        @test get_model_type(reg_manager) == "Regression"
        @test get_initial_condition(reg_manager) === nothing

        # segment_model
        seg_pars = [3.0, 7.0]
        parnames = [:a, :b]
        new_reg_model = segment_model(reg_manager, seg_pars, parnames, 1, 10, nothing)
        @test new_reg_model.time_steps == 10
        @test new_reg_model.params[:b] == 7.0

        # Update initial condition
        reg_sim = simulate_model(reg_model)
        @test update_initial_condition(reg_manager, reg_sim) === nothing
    end
end



# =============================================================================
# Test ObjectiveFunction Module
# =============================================================================

@testset "ObjectiveFunction Tests" begin

    # =========================================================================
    # Test ODE Model
    # =========================================================================
    @testset "ODE Model" begin

        # Define parameters
        p = 0.5
        ic = [1.0]
        tspan = (0.0, 10.0)
        parnames = [:p]
        true_pars = [p]

        # Define model spec
        ode_spec = ODEModelSpec(
           _ModelSimulation.exponential_ode_model,
           Dict(:p => p),
           ic,
           tspan
        )

        observed_df = simulate_model(ode_spec)
        observed_data = reshape(observed_df.state, 1, :)

        # Wrap in manager
        manager = ModelManager(ode_spec)
 
        # Call objective function
        chromosome = copy(true_pars)
        loss = objective_function(
            chromosome,
            Int[],
            0,
            length(parnames),
            parnames,
            manager,
            (obs, sim) -> sum((obs .- sim.state').^2),
            observed_data
        )

        @test loss ≈ 0.0 atol=1e-8

    end

    # =========================================================================
    # Test Difference Model
    # =========================================================================
    @testset "Difference Model" begin
        num_steps = 50
        wind = rand(num_steps)
        temp = rand(num_steps)
        ic = 10.0
  
        # True parameters
        true_pars = [
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        1.0
        ]
        parnames = [:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7]

        # Simulate expected state
        base_model = DifferenceModelSpec(
           _ModelSimulation.example_difference_model,
           Dict(zip(parnames, true_pars)),
           ic,
           num_steps,
           (wind, temp)
        )
        expected_data = simulate_model(base_model)
        observed_data = reshape(expected_data.state, 1, :)

        # Wrap model in manager
        manager = ModelManager(base_model)

        # Chromosome = all params are segment-specific, no constants
        chromosome = copy(true_pars)

        # Evaluate objective
        loss = objective_function(
           chromosome,
           Int[],          # No change points
           2,              
           5,
           parnames,
           manager,
           (obs, sim) -> sum((obs .- sim.state').^2),
           observed_data
           )

        @test loss ≈ 0.0 atol=1e-8
  end

    # =========================================================================
    # Test Regression Model
    # =========================================================================
    @testset "Regression Model" begin

        a, b = 5.0, 2.0
        time_steps = 20
        parnames = [:a, :b]
        true_pars = [a, b]

        # Define regression model
        reg_model = RegressionModelSpec(
           _ModelSimulation.example_regression_model,
           Dict(:a => a, :b => b),
           time_steps
       )

       observed_df = simulate_model(reg_model)
       observed_data = reshape(observed_df.simulated_values, 1, :)

       # Wrap in ModelManager
       manager = ModelManager(reg_model)

       chromosome = copy(true_pars)
       loss = objective_function(
           chromosome,
           Int[],
           0,
           length(parnames),
           parnames,
           manager,
           (obs, sim) -> sum((obs .- sim.simulated_values').^2),
           observed_data
        )

    @test loss ≈ 0.0 atol=1e-8

    end
end