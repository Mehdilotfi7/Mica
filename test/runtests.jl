using Test
using Evolutionary
using LabelledArrays
using DifferentialEquations
using Random
using Mocha

Random.seed!(1234)

# =============================================================================
# Test DataHandler Module
# =============================================================================

# For now this functionality is not available 
#=


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

=#
# =============================================================================
# Test ModelSimulation Module
# =============================================================================

@testset "ModelSimulation Tests" begin

    # ----------------------
    # Test ODE Model
    # ----------------------
    @testset "ODE Model Test" begin
        params = [0.5]
        initial_conditions = [1.0]
        tspan = (0.0, 99.0)
        
        model = ODEModelSpec(exponential_ode_model, params, initial_conditions, tspan)
        result = simulate_model(model)

        @test isa(result, Matrix)       # Output is in a Matrix form
        @test size(result, 2) == 100    # Should have 100 time points
        @test all(result[:,1] .>= 0.0)  # Exponential decay, should be non-negative
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

        @test isa(result, Matrix)
        @test size(result, 1) == num_steps
    end

    # ----------------------
    # Test Regression Model
    # ----------------------
    @testset "Regression Model Test" begin
        params = [2, 1]
        time_steps = 30

        model = RegressionModelSpec(example_regression_model, params, time_steps)
        result = simulate_model(model)

        @test isa(result, Matrix)
        @test size(result, 1) == time_steps
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
        ode_params = [0.5]
        ode_u0 = [1.0]
        tspan = (0.0, 10.0)
        

        base_model = ODEModelSpec(exponential_ode_model, ode_params, ode_u0, tspan)
        manager = ModelManager(base_model)
        new_ode_model = segment_model(manager, ode_params, 0, 10, ode_u0)

        # Check initial condition accessor
        @test get_initial_condition(manager) == ode_u0
        @test get_model_type(manager) == "ODE"
        # segment_model
        @test new_ode_model.params == [0.5]
        @test new_ode_model.tspan == (0.0, 10.0)
        @test new_ode_model.initial_conditions == ode_u0

        # Update initial condition
        sim = simulate_model(base_model)
        @test isa(sim, Matrix)
        @test update_initial_condition(manager, sim) == sim[:,end]
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

        diff_model = DifferenceModelSpec(example_difference_model, pars, ic, num_steps, (wind, temp))
        diff_manager = ModelManager(diff_model)

        # Check initial condition accessor
        @test get_model_type(diff_manager) == "Difference"
        @test get_initial_condition(diff_manager) == ic

        # segment_model
        seg_pars = fill(0.2, 7) # update for Difference
        parnames = [:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7]
        new_diff_model = segment_model(diff_manager, seg_pars, 10, 20, 0.0)
        @test new_diff_model.num_steps == 11
        @test new_diff_model.extra_data[1] == wind[10:20]

        # Update initial condition
        diff_sim = simulate_model(diff_model)
        @test isa(diff_sim, Matrix)
        @test update_initial_condition(diff_manager, diff_sim) == diff_sim[:,end]
    end

    # =========================================================================
    # Test Regression Model
    # =========================================================================
    @testset "Regression Model" begin
        pars = [2, 1]
        time_steps = 30
        reg_model = RegressionModelSpec(example_regression_model, pars, time_steps)
        reg_manager = ModelManager(reg_model)

        #Check initial condition accessor
        @test get_model_type(reg_manager) == "Regression"
        @test get_initial_condition(reg_manager) === nothing

        # segment_model
        seg_pars = [3.0, 7.0]
        parnames = [:a, :b]
        new_reg_model = segment_model(reg_manager, seg_pars, 1, 10, nothing)
        @test new_reg_model.time_steps == 10
        @test new_reg_model.params[2] == 7.0

        # Update initial condition
        reg_sim = simulate_model(reg_model)
        @test update_initial_condition(reg_manager, reg_sim) === nothing
    end
end



# =============================================================================
# Test ObjectiveFunction Module
# =============================================================================

# =========================================================================
# Test extract_parameters
# =========================================================================
@testset "extract_parameters Tests" begin

    p = [0.5, 0.7, 0.9]
    cons_par, seg_par = extract_parameters(p, 1, 1)
    # constant parameter
    @test cons_par == [0.5]
    # first segment parrameter
    @test seg_par[1] == [0.7]
    # second segment parrameter
    @test seg_par[2] == [0.9]

end

# =========================================================================
# Test objective_function
# =========================================================================

@testset "objective_function Tests" begin

    # =========================================================================
    # Test ODE Model
    # =========================================================================
    @testset "ODE Model" begin

        # Define parameters
        p = [0.5]
        ic = [1.0]
        tspan = (0.0, 10.0)
        parnames = (:p)
        true_pars = p

        # Define model spec
        ode_spec = ODEModelSpec(
           exponential_ode_model,
           p,
           ic,
           tspan
        )

        observed_data = simulate_model(ode_spec)

        # Wrap in manager
        manager = ModelManager(ode_spec)
 
        # Call objective function
        chromosome = copy(true_pars)

        loss = objective_function(
            chromosome,
            [],
            parnames,
            0,
            length(parnames),
            manager,
            (obs, sim) -> sum((obs .- sim').^2),
            observed_data
        )

        #@test loss ≈ 0.0 atol=1e-8

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
        parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)

        # Simulate expected state
        base_model = DifferenceModelSpec(
           example_difference_model,
           Dict(zip(parnames, true_pars)),
           ic,
           num_steps,
           (wind, temp)
        )
        expected_data = simulate_model(base_model)
        observed_data = reshape(expected_data, 1, :)

        # Wrap model in manager
        manager = ModelManager(base_model)

        # Chromosome = all params are segment-specific, no constants
        chromosome = copy(true_pars)

        # Evaluate objective
        loss = objective_function(
           chromosome,
           [],          # No change points
           parnames,
           2,              
           5,
           manager,
           (obs, sim) -> sum((obs .- sim').^2),
           observed_data
           )

        @test loss ≈ 0.0 atol=1e-8
  end

    # =========================================================================
    # Test Regression Model
    # =========================================================================
    @testset "Regression Model" begin

        a, b = 5.0, 2.0
        pars = [a, b]
        time_steps = 20
        parnames = (:a, :b)
        true_pars = [a, b]

        # Define regression model
        reg_model = RegressionModelSpec(
           example_regression_model,
           pars,
           time_steps
       )

       observed_data = simulate_model(reg_model)
       observed_data = reshape(observed_data, 1, :)

       # Wrap in ModelManager
       manager = ModelManager(reg_model)

       chromosome = copy(true_pars)
       loss = objective_function(
           chromosome,
           Int[],
           parnames,
           1,
           1,
           manager,
           (obs, sim) -> sum((obs .- sim').^2),
           observed_data
        )

    @test loss ≈ 0.0 atol=1e-8

    end
end

# =============================================================================
# Test ChangePointDetection Module
# =============================================================================

@testset "ChangePointDetection Tests" begin

    # =========================================================================
    # Test ODE Model
    # =========================================================================

    @testset "ODE Model" begin
        # Create synthetic ODE test model
        p = -0.5
        ic = [1.0]
        tspan = (1.0, 100.0)
        parnames = (:p)

        # Define ModelManager for ODE
        ode_spec = ODEModelSpec(
           exponential_ode_model,
           p,
           ic,
           tspan
        )
        manager = ModelManager(ode_spec)
        # Generate synthetic observation
        obs_df = simulate_model(ode_spec)
        data = reshape(obs_df, 1, :)

        # optimize_with_changepoints
        @testset "optimize_with_changepoints" begin
            chromosome = [-0.3]
            bounds = ([-1.0], [0.0])
            CP = Int[]
            n_global = 0
            n_segment_specific = 1
        
            loss_fn(obs, sim) = sum((obs .- sim').^2)
            ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
            crossoverRate=0.6, mutation = gaussian(0.0001))
        
            loss, best = optimize_with_changepoints(
                objective_function,
                chromosome,
                parnames,
                CP,
                bounds,
                ga,
                n_global,
                n_segment_specific,
                manager,
                loss_fn,
                #(obs, sim) -> sum((obs .- sim').^2),
                data
            )
        
            @test isfinite(loss)
            @test isapprox(best[1], -0.5; rtol=1e-1, atol=2e-1)
        end

        # update_bounds!
        @testset "update_bounds!" begin
            chromosome = [-0.5]
            bounds = ([-1.0], [0.0])
            n_global = 0
            n_segment_specific = 1
        
            update_bounds!(
                chromosome,
                bounds,
                n_global,
                n_segment_specific,
                extract_parameters
            )
        
            @test length(chromosome) == 2
            @test length(bounds[1]) == 2
            @test length(bounds[2]) == 2
        end

        # evaluate_segment
        @testset "evaluate_segment" begin
            chromosome = [-0.5, -0.5, -0.5]  
            bounds = ([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0])
            CP = [5]
            a, b = 1, size(data, 2)
            step = 10
            min_length = 10
            pen = 1.0
            n = size(data,2)
        
            loss_fn(obs, sim) = sum((obs .- sim').^2)
            ga = ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
            crossoverRate=0.6, mutation = gaussian(0.0001))
            penalty_fun(p,n) = 1.0 * p * log(n)
        
            x, y = evaluate_segment(
                objective_function,
                a, b, CP, bounds, chromosome, parnames, ga, min_length, step,
                0, 1, n, manager, loss_fn, data, penalty_fun
            )
        
            @test length(x) == length(y)
            @test all(isfinite, x)
        end
                         

    end

end


# =============================================================================
# Test penalty Module
# =============================================================================
