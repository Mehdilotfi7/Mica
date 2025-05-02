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
# Test Model Handling + Objective Function Modules
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

        # Simulate base model
        sim = simulate_model(base_model)
        @test isa(sim, DataFrame)

        # Loss function: squared error to self
        function loss_ode(true_data, pred_data)
            sum((true_data[1] .- pred_data.state).^2)
        end

        # Define data and parameter extraction
        data = [sim.state]
        chrom = [0.5]
        extract(chrom, n_g, n_s) = ([], [[chrom[1]]])

        loss = objective_function(
            chrom,
            Int[], 0, 1, extract, [:p],
            manager,
            simulate_model,
            loss_ode,
            data
        )
        
        @test isapprox(loss, 0.0; atol=1e-4)
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

        base_model = DifferenceModelSpec(_ModelSimulation.example_difference_model, pars, ic, num_steps, (wind, temp))
        manager = ModelManager(base_model)

        # Basic checks
        @test get_model_type(manager) == "Difference"
        @test get_initial_condition(manager) == ic

        sim = simulate_model(base_model)
        @test isa(sim, DataFrame)

        # Loss
        function loss_diff(true_data, pred_data)
            sum((true_data[1] .- pred_data.state).^2)
        end

        data = [sim.state]
        chrom = collect(values(pars))  # [θ1, θ2, ..., θ7]
        extract(chrom, n_g, n_s) = ([], [chrom])

        loss = objective_function(
            chrom,
            Int[], 0, 7, extract, collect(keys(pars)),
            manager,
            simulate_model,
            loss_diff,
            data
        )
        @test isapprox(loss, 0.0; atol=1e-4)
    end

    # =========================================================================
    # Test Regression Model
    # =========================================================================
    @testset "Regression Model" begin
        pars = Dict(:a => 2.0, :b => 1.0)
        time_steps = 30
        base_model = RegressionModelSpec(_ModelSimulation.example_regression_model, pars, time_steps)
        manager = ModelManager(base_model)

        @test get_model_type(manager) == "Regression"
        @test get_initial_condition(manager) === nothing

        sim = simulate_model(base_model)
        @test isa(sim, DataFrame)

        # Loss
        function loss_reg(true_data, pred_data)
            sum((true_data[1] .- pred_data.simulated_values).^2)
        end

        data = [sim.simulated_values]
        chrom = [2.0, 1.0]
        extract(chrom, n_g, n_s) = ([], [chrom])

        loss = objective_function(
            chrom,
            Int[], 0, 2, extract, [:a, :b],
            manager,
            simulate_model,
            loss_reg,
            data
        )
        @test isapprox(loss, 0.0; atol=1e-4)
    end
end



# =============================================================================
# Test ObjectiveFunction Module
# =============================================================================

@testset "ObjectiveFunction Tests" begin

    # Dummy loss function (mean squared error)
    mse(y, ŷ) = mean((y .- ŷ).^2)

    @testset "segment_loss 1D test" begin
        true_data = [1.0, 2.0, 3.0]
        predicted_data = [1.0, 2.5, 2.5]
        loss = segment_loss(true_data, predicted_data, mse)
        @test isapprox(loss, 0.1666; atol=1e-3)
    end

    @testset "extract_parameters test" begin
        chrom = [1.0, 2.0, 3.0, 4.0, 5.0]
        global_parameters, segment_parameters = extract_parameters(chrom, 2, 3)
        @test global_parameters == [1.0, 2.0]
        @test segment_parameters == [[3.0, 4.0, 5.0]]
    end

    @testset "objective_function test - regression" begin
        model_func = example_regression_model
        sim_func = simulate_model

        parnames = [:a, :b]
        chrom = [2.0, 5.0]  # slope 2, intercept 5
        CP = Int[]  # no change points
        n_global = 0
        n_segment_specific = 2
        initial_conditions = nothing
        num_steps = 10
        tspan = nothing
        extra_data = nothing
        time = 10

        data = model_func(Dict(:a => 2.0, :b => 5.0), time)
        data_CP = data.simulated_values  # pass only values to loss function

        loss_fn = mse
        loss = objective_function(
            chrom, CP, n_global, n_segment_specific, extract_parameters,
            parnames, model_func, sim_func, loss_fn, segment_loss, data_CP;
            num_steps=num_steps,
            initial_conditions=initial_conditions,
            extra_data=extra_data,
            compare_variables=nothing
        )
        @test isapprox(loss, 0.0; atol=1e-5)
    end

end