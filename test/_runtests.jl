using Test
using TSCPDetector.ModelSimulation  # Assuming TSCPDetector is your package

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