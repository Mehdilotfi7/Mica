using Test
using TSCPDetector.ModelSimulation  # Assuming TSCPDetector is your package

# =============================================================================
# Test Example Models
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

        #@test isa(result, DataFrame)
        #@test size(result, 1) == 100  # Should have 100 time points
        #@test :time in names(result)
        #@test :state in names(result)
        #@test all(result.state .>= 0.0)  # Exponential decay, should be non-negative
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
        @test :time in names(result)
        @test :state in names(result)
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
        @test :time in names(result)
        @test :simulated_values in names(result)
        @test all(result.simulated_values .== params[:a] .* result.time .+ params[:b])
    end

end
