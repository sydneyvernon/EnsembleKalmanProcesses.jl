using Distributions
using LinearAlgebra
using Random
using JLD2

#using Plots
#using PythonPlot
using Plots; gr()
using Plots.PlotMeasures

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses
fig_save_directory = @__DIR__

# the package to define the function distributions
import GaussianRandomFields # we wrap this so we don't want to use "using"
const GRF = GaussianRandomFields

# We include the forward solver here
include("GModel.jl")



function rk4(f::F, y0::Array{Float64, 1}, t0::Float64, t1::Float64, h::Float64; inplace::Bool = true) where {F}
    y = y0
    n = round(Int, (t1 - t0) / h)
    t = t0
    if ~inplace
        hist = zeros(n, length(y0))
    end
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if ~inplace
            hist[i, :] = y
        end
        t = t0 + i * h
    end
    if ~inplace
        return hist
    else
        return y
    end
end

function lorenz96(t, u, p)
    N = p["N"]
    F = 8

    du = similar(u)

    for i in 1:N
        du[i] = (u[mod(i + 1, 1:N)] - u[mod(i - 2, 1:N)]) * u[mod(i - 1, 1:N)] - u[i] + F
    end

    return copy(du)
end


function expsin_main(N_ens_, N_iter_, N_trials_, process; localizer=nothing)
    ## exp sin setup
    rng_seed = 41
    rng = Random.MersenneTwister(rng_seed)
    dt = 0.01
    trange = 0:dt:(2 * pi + dt)
    function model(amplitude, vert_shift)
        phi = 2 * pi * rand(rng)
        return exp.(amplitude * sin.(trange .+ phi) .+ vert_shift)
    end

    function G(u)
        theta, vert_shift = u
        sincurve = model(theta, vert_shift)
        return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
    end

    USE_SCHEDULER = false
    scheduler_def = DataMisfitController(on_terminate = "continue")

    dim_output = 2
    Γ = 0.01 * I
    noise_dist = MvNormal(zeros(dim_output), Γ)
    theta_true = [1.0, 0.8]
    y = G(theta_true) .+ rand(noise_dist)

    # We define a variety of prior distributions so we can study
    # the effectiveness of accelerators on this problem.

    prior_u1 = constrained_gaussian("amplitude", 2, 0.1, 0, 10)
    prior_u2 = constrained_gaussian("vert_shift", 0, 0.5, -10, 10)
    prior = combine_distributions([prior_u1, prior_u2])

    # To compare the two EKI methods, we will average over several trials, 
    # allowing the methods to run with different initial ensembles and noise samples.
    N_ens = N_ens_
    N_iterations = N_iter_
    N_trials = N_trials_

    ## Solving the inverse problem

    # Preallocate so we can track and compare convergences of the methods
    all_convs = zeros(N_trials, N_iterations)
    all_convs_acc = zeros(N_trials, N_iterations)
    all_convs_acc_cs = zeros(N_trials, N_iterations)
    theta_history = zeros(N_trials, N_iterations)

    for trial in 1:N_trials
        # We now generate the initial ensemble and set up two EKI objects, one using an accelerator, 
        # to compare convergence.
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        if USE_SCHEDULER
            scheduler = deepcopy(scheduler_def)
            scheduler_acc = deepcopy(scheduler_def)
            scheduler_acc_cs = deepcopy(scheduler_def)
        else
            scheduler = DefaultScheduler()
            scheduler_acc = DefaultScheduler()
            scheduler_acc_cs = DefaultScheduler()
        end
        prior2 = deepcopy(prior)
        prior3 = deepcopy(prior) # i think redundant
        if process == "Inversion"
            ensemble_kalman_process =
                EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler = scheduler)
            ensemble_kalman_process_acc = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion();accelerator = NesterovAccelerator(),rng = rng,scheduler = scheduler_acc,
                #localization_method=localizer,
            )
            ensemble_kalman_process_acc_cs = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion();accelerator = ConstantStepNesterovAccelerator(),rng = rng,scheduler = scheduler_acc_cs,
                #localization_method=localizer,
            )
        elseif process == "Unscented"
            ensemble_kalman_process =
                EKP.EnsembleKalmanProcess(y, Γ, Unscented(prior); rng = rng, scheduler = scheduler)
            ensemble_kalman_process_acc = EKP.EnsembleKalmanProcess(y,Γ,Unscented(prior2);accelerator = NesterovAccelerator(),rng = rng,scheduler = scheduler_acc,
            )
            ensemble_kalman_process_acc_cs = EKP.EnsembleKalmanProcess(y,Γ,Unscented(prior3);accelerator = ConstantStepNesterovAccelerator(),rng = rng,scheduler = scheduler_acc_cs,
            )
        elseif process == "TransformInversion"
            ensemble_kalman_process =
                EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, TransformInversion(inv(Γ)); rng = rng, scheduler = scheduler)
            ensemble_kalman_process_acc = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,TransformInversion(inv(Γ));accelerator = NesterovAccelerator(),rng = rng,scheduler = scheduler_acc,
            )
            ensemble_kalman_process_acc_cs = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,TransformInversion(inv(Γ));accelerator = ConstantStepNesterovAccelerator(),rng = rng,scheduler = scheduler_acc_cs,
            )
        else
            print("!! invalid process !!")
        end

        global convs = zeros(N_iterations)
        global convs_acc = zeros(N_iterations)
        global convs_acc_cs = zeros(N_iterations)

        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.
        # We perform the inversion in parallel to compare the two EKI methods.
        for i in 1:N_iterations
            # G_ens = G(get_ϕ_final(prior, ensemble_kalman_process))
            # G_ens_acc = G(get_ϕ_final(prior2, ensemble_kalman_process_acc))
            # G_ens_acc_cs = G(get_ϕ_final(prior3, ensemble_kalman_process_acc_cs))

            params_i = get_ϕ_final(prior, ensemble_kalman_process)
            params_i_acc = get_ϕ_final(prior2, ensemble_kalman_process_acc)
            params_i_acc_cs = get_ϕ_final(prior3, ensemble_kalman_process_acc_cs)
            G_ens = hcat([G(params_i[:, i]) for i in 1:size(params_i, 2)]...)
            G_ens_acc = hcat([G(params_i_acc[:, i]) for i in 1:size(params_i_acc, 2)]...)
            G_ens_acc_cs = hcat([G(params_i_acc_cs[:, i]) for i in 1:size(params_i_acc_cs, 2)]...)

            EKP.update_ensemble!(ensemble_kalman_process, G_ens) #deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc, G_ens_acc) #deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc_cs, G_ens_acc_cs) #deterministic_forward_map = false)

            convs[i] =  get_error(ensemble_kalman_process)[end]
            convs_acc[i] =  get_error(ensemble_kalman_process_acc)[end]
            convs_acc_cs[i] =  get_error(ensemble_kalman_process_acc_cs)[end]
            theta_history[trial,i] = ensemble_kalman_process_acc.accelerator.θ_prev
        end
        all_convs[trial, :] = convs
        all_convs_acc[trial, :] = convs_acc
        all_convs_acc_cs[trial, :] = convs_acc_cs
    end
    return all_convs, all_convs_acc, all_convs_acc_cs, theta_history
end


function lorenz_main(N_ens_, N_iter_, N_trials_, process; localizer=nothing)
    D = 20
    USE_SCHEDULER = false
    scheduler_def = DataMisfitController(on_terminate = "continue")

    lorenz96_sys = (t, u) -> lorenz96(t, u, Dict("N" => D))

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)
    dt = 0.05
    y0 = rk4(lorenz96_sys, randn(D), 0.0, 1000.0, dt)

    # Lorenz96 initial condition problem - Section 6.3 of Tong and Morzfeld (2022)
    G(u) = mapslices((u) -> rk4(lorenz96_sys, u, 0.0, 0.4, dt), u, dims = 1)
    p = D
    

    N_ens = N_ens_
    N_iter = N_iter_
    N_trials = N_trials_
    errs = zeros(N_trials, N_iter)
    errs_acc = zeros(N_trials, N_iter)
    errs_acc_cs = zeros(N_trials, N_iter)
    theta_history = zeros(N_trials, N_iter)

    for trial in 1:N_trials
        # Generate random truth
        y = y0 + randn(D)
        Γ = 1.0 * I

        #### Define prior information on parameters
        priors = map(1:p) do i
            constrained_gaussian(string("u", i), 0.0, 1.0, -Inf, Inf)
        end
        prior = combine_distributions(priors)

        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        if USE_SCHEDULER
            scheduler_van = deepcopy(scheduler_def)
            scheduler_acc = deepcopy(scheduler_def)
            scheduler_acc_cs = deepcopy(scheduler_def)
        else
            scheduler_van = DefaultScheduler()
            scheduler_acc = DefaultScheduler()
            scheduler_acc_cs = DefaultScheduler()
        end
        # We create 3 EKP Inversion objects to compare acceleration.
        prior2 = deepcopy(prior)
        prior3 = deepcopy(prior) # redundant ??
        if process=="Inversion"
            ekiobj_vanilla =
                EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler = scheduler_van)
            ekiobj_acc = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion();rng = rng,accelerator = NesterovAccelerator(),scheduler = scheduler_acc,)
            ekiobj_acc_cs = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion(); rng = rng,accelerator = ConstantStepNesterovAccelerator(),scheduler = scheduler_acc_cs,)
            if isnothing(localizer)==false
                ekiobj_vanilla =
                EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler = scheduler_van, localization_method=deepcopy(localizer))
                ekiobj_acc = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion();rng = rng,accelerator = NesterovAccelerator(),scheduler = scheduler_acc,localization_method=deepcopy(localizer))
                ekiobj_acc_cs = EKP.EnsembleKalmanProcess(initial_ensemble,y,Γ,Inversion(); rng = rng,accelerator = ConstantStepNesterovAccelerator(),scheduler = scheduler_acc_cs,localization_method=deepcopy(localizer))
            end
        elseif process=="Unscented"
            ekiobj_vanilla =
                EKP.EnsembleKalmanProcess(y, Γ, Unscented(prior); rng = rng, scheduler = scheduler_van)
            ekiobj_acc = EKP.EnsembleKalmanProcess(y,Γ,Unscented(prior2);rng = rng,accelerator = NesterovAccelerator(),scheduler = scheduler_acc,)
            ekiobj_acc_cs = EKP.EnsembleKalmanProcess(y,Γ,Unscented(prior3); rng = deepcopy(rng),accelerator = ConstantStepNesterovAccelerator(),scheduler = scheduler_acc_cs,)
        end

        err = zeros(N_iter)
        err_acc = zeros(N_iter)
        err_acc_cs = zeros(N_iter)
        for i in 1:N_iter
            g_ens_vanilla = G(get_ϕ_final(prior, ekiobj_vanilla))
            EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla) # , deterministic_forward_map = true)
            g_ens_acc = G(get_ϕ_final(prior2, ekiobj_acc))
            EKP.update_ensemble!(ekiobj_acc, g_ens_acc) # deterministic_forward_map = true)
            g_ens_acc_cs = G(get_ϕ_final(prior3, ekiobj_acc_cs))
            EKP.update_ensemble!(ekiobj_acc_cs, g_ens_acc_cs) #, deterministic_forward_map = true)

            theta_history[trial,i] = ekiobj_acc.accelerator.θ_prev
        end
        errs[trial, :] = get_error(ekiobj_vanilla)
        errs_acc[trial, :] = get_error(ekiobj_acc)
        errs_acc_cs[trial, :] = get_error(ekiobj_acc_cs)
    end
    return errs, errs_acc, errs_acc_cs, theta_history
end



function darcy_main(N_ens_, N_iter_, N_trials_, process)
        # Set a random seed.
        seed = 100234
        rng = Random.MersenneTwister(seed)
    
        USE_SCHEDULER = false
        scheduler_def = DataMisfitController(on_terminate = "continue")
    
        # Define the spatial domain and discretization 
        dim = 2
        N, L = 80, 1.0
        pts_per_dim = LinRange(0, L, N)
        obs_ΔN = 10
    
        # To provide a simple test case, we assume that the true function parameter is a particular sample from the function space we set up to define our prior. More precisely we choose a value of the truth that doesnt have a vanishingly small probability under the prior defined by a probability distribution over functions; here taken as a family of Gaussian Random Fields (GRF). The function distribution is characterized by a covariance function - here a Matern kernel which assumes a level of smoothness over the samples from the distribution. We define an appropriate expansion of this distribution, here based on the Karhunen-Loeve expansion (similar to an eigenvalue-eigenfunction expansion) that is truncated to a finite number of terms, known as the degrees of freedom (`dofs`). The `dofs` define the effective dimension of the learning problem, decoupled from the spatial discretization. Explicitly, larger `dofs` may be required to represent multiscale functions, but come at an increased dimension of the parameter space and therefore a typical increase in cost and difficulty of the learning problem.
        smoothness = 1.0
        corr_length = 0.25
        dofs = 50
    
        grf = GRF.GaussianRandomField(
            GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
            GRF.KarhunenLoeve(dofs),
            pts_per_dim,
            pts_per_dim,
        )
    
        # We define a wrapper around the GRF, and as the permeability field must be positive we introduce a domain constraint into the function distribution. Henceforth, the GRF is interfaced in the same manner as any other parameter distribution with regards to interface.
        pkg = GRFJL()
        distribution = GaussianRandomFieldInterface(grf, pkg) # our wrapper from EKP
        domain_constraint = bounded_below(0) # make κ positive
        pd = ParameterDistribution(
            Dict("distribution" => distribution, "name" => "kappa", "constraint" => domain_constraint),
        ) # the fully constrained parameter distribution
    
        # Now we have a function distribution, we sample a reasonably high-probability value from this distribution as a true value (here all degrees of freedom set with `u_{\mathrm{true}} = -0.5`). We use the EKP transform function to build the corresponding instance of the ``\kappa_{\mathrm{true}}``.
        u_true = -1.5 * ones(dofs, 1) # the truth parameter
        κ_true = transform_unconstrained_to_constrained(pd, u_true) # builds and constrains the function.  
        κ_true = reshape(κ_true, N, N)
    
        # Now we generate the data sample for the truth in a perfect model setting by evaluating the the model here, and observing it by subsampling in each dimension every `obs_ΔN` points, and add some observational noise
        darcy = Setup_Param(pts_per_dim, obs_ΔN, κ_true)
        h_2d = solve_Darcy_2D(darcy, κ_true)
        y_noiseless = compute_obs(darcy, h_2d)
        obs_noise_cov = 0.05^2 * I(length(y_noiseless)) * (maximum(y_noiseless) - minimum(y_noiseless))
    
        # Now we set up the Bayesian inversion algorithm. The prior we have already defined to construct our truth
        prior = pd
    
        # We define some algorithm parameters, here we take ensemble members larger than the dimension of the parameter space
        N_ens = N_ens_ #dofs + 2    # number of ensemble members
        N_iter = N_iter_         # number of EKI iterations
        N_trials = N_trials_       # number of trials 
    
        errs = zeros(N_trials, N_iter)
        errs_acc = zeros(N_trials, N_iter)
        errs_acc_cs = zeros(N_trials, N_iter)
        theta_history = zeros(N_trials, N_iter)
    
        for trial in 1:N_trials
            truth_sample = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))

            if USE_SCHEDULER
                scheduler = scheduler_def
            else
                scheduler = DefaultScheduler(0.1)
            end
    
            # We sample the initial ensemble from the prior, and create three EKP objects to 
            # perform EKI algorithm using three different acceleration methods.
            initial_params = construct_initial_ensemble(rng, prior, N_ens)
            prior2 = deepcopy(prior)
            prior3 = deepcopy(prior)
            if process=="Inversion"
                ekiobj = EKP.EnsembleKalmanProcess(initial_params,truth_sample,obs_noise_cov,Inversion(),scheduler = deepcopy(scheduler),)
                ekiobj_acc = EKP.EnsembleKalmanProcess(initial_params,truth_sample,obs_noise_cov,Inversion(),accelerator = NesterovAccelerator(),scheduler = deepcopy(scheduler),)
                ekiobj_acc_cs = EKP.EnsembleKalmanProcess(initial_params,truth_sample,obs_noise_cov,Inversion(),accelerator = ConstantStepNesterovAccelerator(),scheduler = deepcopy(scheduler),)
            elseif process=="Unscented"
                ekiobj = EKP.EnsembleKalmanProcess(truth_sample,obs_noise_cov,Unscented(prior);scheduler = deepcopy(scheduler),rng=rng)
                ekiobj_acc = EKP.EnsembleKalmanProcess(truth_sample,obs_noise_cov,Unscented(prior2);accelerator = NesterovAccelerator(),scheduler = deepcopy(scheduler),rng=rng)
                ekiobj_acc_cs = EKP.EnsembleKalmanProcess(truth_sample,obs_noise_cov,Unscented(prior3);accelerator = ConstantStepNesterovAccelerator(),scheduler = deepcopy(scheduler),rng=rng)
            end
            # Run EKI algorithm, recording parameter error after each iteration.
            err = zeros(N_iter)
            err_acc = zeros(N_iter)
            err_acc_cs = zeros(N_iter)
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekiobj)
                params_i_acc = get_ϕ_final(prior2, ekiobj_acc)
                params_i_acc_cs = get_ϕ_final(prior3, ekiobj_acc_cs)
    
                g_ens = run_G_ensemble(darcy, params_i)
                g_ens_acc = run_G_ensemble(darcy, params_i_acc)
                g_ens_acc_cs = run_G_ensemble(darcy, params_i_acc_cs)
                # params_i = get_ϕ_final(prior, ekiobj)
                # params_i_acc = get_ϕ_final(prior2, ekiobj_acc)
                # params_i_acc_cs = get_ϕ_final(prior3, ekiobj_acc_cs)
                # g_ens = hcat([G(params_i[:, i]) for i in 1:size(params_i, 2)]...)
                # g_ens_acc = hcat([G(params_i_acc[:, i]) for i in 1:size(params_i_acc, 2)]...)
                # g_ens_acc_cs = hcat([G(params_i_acc_cs[:, i]) for i in 1:size(params_i_acc_cs, 2)]...)
    
                EKP.update_ensemble!(ekiobj, g_ens) #, deterministic_forward_map = true)
                EKP.update_ensemble!(ekiobj_acc, g_ens_acc) #, deterministic_forward_map = true)
                EKP.update_ensemble!(ekiobj_acc_cs, g_ens_acc_cs) #, deterministic_forward_map = true)
    
                theta_history[trial,i] = ekiobj_acc.accelerator.θ_prev
                err[i] = get_error(ekiobj)[end]
                errs[trial, :] = err
                err_acc[i] = get_error(ekiobj_acc)[end]
                errs_acc[trial, :] = err_acc
                err_acc_cs[i] = get_error(ekiobj_acc_cs)[end]
                errs_acc_cs[trial, :] = err_acc_cs
            end
        end
        return errs, errs_acc, errs_acc_cs, theta_history
end



"""
Plot making functions
"""
function fig1(N_ens, N_iter, N_trials, errs, errs_acc, errs_acc_cs, theta_history, filename, plotname, LOGSCALE)
    if LOGSCALE
        errs = log.(errs)
        errs_acc = log.(errs_acc)
        errs_acc_cs = log.(errs_acc_cs)
    end
    convplot = plot(1:(N_iter), mean(errs, dims = 1)[:], color = :black, label = "No acceleration", left_margin = [5mm 0mm], bottom_margin = 10px)
    plot!(1:(N_iter), mean(errs_acc, dims = 1)[:], color = :blue, label = "Nesterov")
    plot!(1:(N_iter), mean(errs_acc_cs, dims = 1)[:], color = :red, label = "Nesterov (original update)")
    title!(plotname)
    ## ERROR BARS
    plot!(
            1:(N_iter),
            (mean(errs, dims = 1)[:] + std(errs, dims = 1)[:] / sqrt(N_trials)),
            color = :black,
            ls = :dash,
            label="",
    )
    plot!(
            1:(N_iter),
            (mean(errs, dims = 1)[:] - std(errs, dims = 1)[:] / sqrt(N_trials)),
            color = :black,
            ls = :dash,
            label = "",
    )
    plot!(
            1:(N_iter),
            (mean(errs_acc, dims = 1)[:] + std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
            color = :blue,
            ls = :dash,
            label="",
    )
    plot!(
            1:(N_iter),
            (mean(errs_acc, dims = 1)[:] - std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
            color = :blue,
            ls = :dash,
            label = "",
    )
    plot!(
            1:(N_iter),
            (mean(errs_acc_cs, dims = 1)[:] + std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
            color = :red,
            ls = :dash,
            label=""
    )
    plot!(
            1:(N_iter),
            (mean(errs_acc_cs, dims = 1)[:] - std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
            color = :red,
            ls = :dash,
            label = "",
    )
    xlabel!("Iteration")
    if LOGSCALE
        ylabel!("log(Error)")
    else
        ylabel!("Error")
    end

    coef_1 = ones(N_iter).-3*ones(N_iter)./((1:N_iter).+2) ## corresponds to "constant step" Nesterov
    coef_2 = zeros(N_iter)
    theta_h = mean(theta_history, dims=1)  # the standard dev of these values between trials is super small
    for i in 1:(N_iter-1)
        coef_2[i+1] = theta_h[i+1]*(theta_h[i].^-1 - 1)
    end
    coeffplot = plot(1:N_iter,  coef_2, color=:blue, label="Nesterov")
    plot!(1:N_iter,  coef_1, color=:red, label="Nesterov (original update)")
    xlabel!("Iteration")
    title!("Momentum coefficient for various methods")
    
    #savefig(coeffplot, fig_save_directory*"/"*filename*"_coeff_plot")
    savefig(convplot, fig_save_directory*"/"*filename*"_conv_plot")
end



"""
Per example (exp-sin, Lorenz, Darcy): One experiment using EKI and constant timestep,
one panel comparing the error using different Nesterov coefficients. Add a final panel
comparing the Nestorov momentum ”coefficient” values (i.e. just plot 1-3/(k+2), and plot
the θk+1(θ−1 
k − 1)). Compare (at fixed timestep) the 1 − 3k−1, the θk(θ−1
k−1 − 1) Su et al.
(2016), and any other variants we find that display the limiting 1−3k−1 +O(k−2) behaviour
for large k (this is where the theory defines acceleration). [Maybe also look at constant?]
"""

N_ens=20
N_iter=20
N_trials=200
lor_errs, lor_errs_acc, lor_errs_acc_cs, lor_theta_history = lorenz_main(N_ens,N_iter,N_trials,"Inversion")
fig1(N_ens,N_iter,N_trials,lor_errs,lor_errs_acc,lor_errs_acc_cs,lor_theta_history,"lorenz","EKI convergence on Lorenz96",false)

# N_ens=50
# N_iter=20
# N_trials=10
# dar_errs, dar_errs_acc, dar_errs_acc_cs, dar_theta_history = darcy_main(N_ens,N_iter,N_trials,"Inversion")
# fig1(N_ens,N_iter,N_trials,dar_errs,dar_errs_acc,dar_errs_acc_cs,dar_theta_history,"darcy_50ens_20iter_10trials","Darcy",true)


#UKI TESTS
# N_ens=20
# N_iter=20
# N_trials=200
# es_errs, es_errs_acc, es_errs_acc_cs, es_theta_history = expsin_main(N_ens,N_iter,N_trials,"Unscented")
# fig1(N_ens,N_iter,N_trials,es_errs,es_errs_acc,es_errs_acc_cs,es_theta_history,"expsin_UKI","UKI convergence on Exp Sin",true)

# N_ens=20
# N_iter=20
# N_trials=200
# es_errs, es_errs_acc, es_errs_acc_cs, es_theta_history = lorenz_main(N_ens,N_iter,N_trials,"Unscented")
# fig1(N_ens,N_iter,N_trials,es_errs,es_errs_acc,es_errs_acc_cs,es_theta_history,"lorenz_UKI","UKI convergence on Lorenz96",true)

# N_ens=20
# N_iter=20
# N_trials=10
# es_errs, es_errs_acc, es_errs_acc_cs, es_theta_history = darcy_main(N_ens,N_iter,N_trials,"Unscented")
# fig1(N_ens,N_iter,N_trials,es_errs,es_errs_acc,es_errs_acc_cs,es_theta_history,"darcy_UKI","UKI convergence on Darcy",true)


# LOCALIZER TOGGLE DOESNT WORK YET



N_iter=20
N_trials=500
# small_errs, small_errs_acc, small_errs_acc_cs, small_theta_history = lorenz_main(5,N_iter,N_trials,"Inversion";localizer=SEC(0.1))
# #sm_errs, sm_errs_acc, sm_errs_acc_cs, sm_theta_history = lorenz_main(20,N_iter,N_trials,"Inversion")
# mid_errs, mid_errs_acc, mid_errs_acc_cs, mid_theta_history = lorenz_main(20,N_iter,N_trials,"Inversion")
# big_errs, big_errs_acc, big_errs_acc_cs, big_theta_history = lorenz_main(50,N_iter,N_trials,"Inversion")
# enssize = plot(1:N_iter, mean((small_errs), dims=1)[:], color=:black, ls=:dot, label="")
# plot!(1:N_iter, mean((mid_errs), dims=1)[:], color=:black, ls=:dash, label="")
# plot!(1:N_iter, mean((big_errs), dims=1)[:], color=:black, label="")

# plot!(1:N_iter, mean((small_errs_acc), dims=1)[:], color=:blue, ls=:dot, label="")
# plot!(1:N_iter, mean((mid_errs_acc), dims=1)[:], color=:blue, ls=:dash, label="")
# plot!(1:N_iter, mean((big_errs_acc), dims=1)[:], color=:blue, label="")

# plot!(1:N_iter, mean((small_errs_acc_cs), dims=1)[:], color=:red, ls=:dot, label="")
# plot!(1:N_iter, mean((mid_errs_acc_cs), dims=1)[:], color=:red, ls=:dash, label="")
# plot!(1:N_iter, mean((big_errs_acc_cs), dims=1)[:], color=:red, label="")

# title!("EKI convergence on Lorenz96 with variable N_ens")
# ylabel!("Error")
# savefig(enssize, fig_save_directory*"/ensemble_size_lorenz_5_20_50")
# savefig(enssize, fig_save_directory*"/ensemble_size_lorenz_5_20_50.pdf")