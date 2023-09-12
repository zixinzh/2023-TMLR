clear
clc
close all

warning('off')


%% 1.initialize
T0 = 4;
T = 10^T0;
no_seed = 1;
seed_range = 1:no_seed;  
disp(['no_seed:', num2str(no_seed) ]);
%%%
reward_distr = 'bern'; % 'gaussStd', 'bern'
disp(['reward_distr:', reward_distr ]);

if strcmp( reward_distr, 'bern' )
    w_para_range = [ 0.5     0.5  0.5;
        0.05    0.1  0.2 ];
elseif strcmp( reward_distr, 'gaussStd' )
    w_para_range = [ 0.5  0.5  0.5;
        0.1  0.2  0.4 ];
end
%%%
L_range = [ 64  128  ];  % no. of all arms



timetable = zeros(1, no_seed);

algo_short_name = {'BoBW_lilUCB', 'UCB_alpha'  };
algo_folder_name = {'1 BoBW_lilUCB', '2 UCB_alpha'   };
algo_print_name = {'1_BoBW_lilUCB', '2_UCB_alpha' };
%%%%%%%%%%%%%%%%%%%%%%%
sigma = 1/2;
beta = exp(1);
varepsilon = 0.01;
BOB_omega_range = 10.^[ -7 -4 -1 ];
%
num_BOB_omega = length( BOB_omega_range );
BOB_alg_para_set_range = [ BOB_omega_range; [ sigma varepsilon beta ]'*ones( 1, num_BOB_omega ) ];
%%%%%%%%%%%%%%%%%%%%%%%
delta = 10^(-2);
alpha_range = [ 1.5 3 6];
%
UCB_alpha_alg_para_set_range = [ alpha_range; [ sigma delta]'*ones( 1, length(alpha_range) ) ];



cum_regret_exp_all = zeros(no_seed,1);
correct_ratio_all = -1*ones(no_seed,1);
t_stop_all = -1*ones(no_seed,1);
flag_terminate_all = -1*ones(no_seed,1);


time_name = ['T=', '10e', num2str(T0)];

correct_count = 0;
fail_terminate_count = 0;



disp('algo_name T std_T_stop L w_opt w_gap_para alg_para_set correct_ratio fail_terminate_count mean_cum_regret_exp std_cum_regret_exp run_time');

for algo_no = [1 2 ]
    
    folder_name = [ 'fixConf ', time_name,...
        ' ', algo_short_name{algo_no}, '_', num2str(no_seed)  ];
    if exist(folder_name, 'dir')==0
        mkdir(folder_name);
        mkdir([folder_name, '_', num2str(no_seed) ]);
    end
    
    for L_ind = 1:length(L_range)
        L = L_range(L_ind);
        for w_para_ind = 1:size(w_para_range,2)
            w_opt = w_para_range(1,w_para_ind);
            w_gap_para = w_para_range(2,w_para_ind);
            if algo_no == 1
                alg_para_set_range = BOB_alg_para_set_range;
            end
            if algo_no == 2
                alg_para_set_range = UCB_alpha_alg_para_set_range;
            end
            
            for alg_para_set_ind = 1: size( alg_para_set_range, 2 )
                alg_para_set = alg_para_set_range(:,alg_para_set_ind);
                
                if algo_no == 1
                    simulation_name = [ algo_print_name{algo_no}, ' T=', num2str(T), ' L=', num2str(L),  ...
                        ' w_opt=', num2str(w_opt), ' w_gap_para=', num2str(w_gap_para), ' reward_distr=', reward_distr,...
                        ' alg_para_set=', num2str(alg_para_set'), ...
                        '_algo ', num2str(algo_no)];
                end
                if algo_no == 2
                    simulation_name = [ algo_print_name{algo_no}, ' L=', num2str(L),  ...
                        ' w_opt=', num2str(w_opt), ' w_gap_para=', num2str(w_gap_para), ' reward_distr=', reward_distr,...
                        ' alg_para_set=', num2str(alg_para_set'), ...
                        '_algo ', num2str(algo_no)];
                end
                
                for seed_index = 1:no_seed
                    seed_val = seed_range(seed_index);
                    rng(seed_val);
                    %% 2.run algorithm
                    tstart = tic;
                    if algo_no == 1
                        [ iout, corr_flag, ...
                            regret, cum_regret, regret_exp, cum_regret_exp, ...
                            w, index_opt, ...
                            W_sel, reward_opt ]...
                            = BoBW_lilUCB_to_submit( T, L, w_opt, w_gap_para, reward_distr, alg_para_set  );
                    end
                    if algo_no == 2
                        T_max = floor(10*T);
                        [ iout, corr_flag, ...
                            regret, cum_regret, regret_exp, cum_regret_exp, ...
                            w, index_opt, ...
                            W_sel, reward_opt,...
                            t, flag_terminate ]...
                            = UCB_alpha_to_submit( T, L, w_opt, w_gap_para, reward_distr, alg_para_set  );
                    end
                    
                    timetable_tmp = toc(tstart);
                    timetable(seed_index) = timetable_tmp;
                    
                    %% 3.seed save
                    if algo_no == 1
                        save([ folder_name, '_', num2str(no_seed), '/seed=', num2str(seed_val), ' ', simulation_name, '.mat'],  ...
                            'iout', 'corr_flag', ...
                            'regret', 'cum_regret', 'regret_exp', 'cum_regret_exp', ...
                            'w', 'index_opt', ...
                            'W_sel', 'reward_opt' );
                        cum_regret_exp_all(seed_index,:) = cum_regret_exp(end);
                    end
                    if algo_no == 2
                        save([ folder_name, '_', num2str(no_seed), '/seed=', num2str(seed_val), ' ', simulation_name, '.mat'],  ...
                            'iout', 'corr_flag', ...
                            'regret', 'cum_regret', 'regret_exp', 'cum_regret_exp', ...
                            'w', 'index_opt', ...
                            'W_sel', 'reward_opt',...
                            't', 'flag_terminate' );
                        t_stop_all(seed_index) = t;
                        flag_terminate_all(seed_index) = flag_terminate;
                        cum_regret_exp_all(seed_index,:) = cum_regret_exp(t);
                    end
                    %% 4.save expected cumulative regret in a matrix
                    correct_ratio_all(seed_index) = corr_flag;
                    if corr_flag == 1
                        correct_count = correct_count + 1;
                    end
                end
                
                %% 5. save 20 sets of data for each setting of K, L and w_gap
                mean_regret = mean(cum_regret_exp_all);
                std_regret = std(cum_regret_exp_all);
                ave_time = mean(timetable);
                if algo_no == 1
                    save([ folder_name, '/',simulation_name, '.mat'], ...
                        'T', 'correct_ratio_all', 'cum_regret_exp_all',...
                        'mean_regret', 'std_regret', 'timetable', 'ave_time');
                    
                    mean_t_stop = T;
                    std_t_stop = 0;
                    fail_terminate_count = 0;
                end
                if algo_no == 2
                    mean_t_stop = mean(t_stop_all);
                    std_t_stop = std(t_stop_all);
                    save([ folder_name, '/',simulation_name, '.mat'], ...
                        'correct_ratio_all', 'cum_regret_exp_all', 't_stop_all', 'flag_terminate_all',...
                        'mean_regret', 'std_regret', 'mean_t_stop', 'std_t_stop', 'timetable', 'ave_time');
                    fail_terminate_count = no_seed - sum(flag_terminate_all);
                end
                
                disp([ algo_print_name{algo_no}, ' ', num2str(mean_t_stop), ' ', num2str(std_t_stop),' ',...
                    num2str(L), ' ', num2str(w_opt), ' ',...
                    num2str(w_gap_para), ' ', num2str(alg_para_set'), ' ' ...
                    num2str( correct_count/no_seed ), ' ', num2str(fail_terminate_count),' ', ...
                    num2str( mean_regret(end) ), ' ', num2str(std_regret(end)), ' ',...
                    num2str(ave_time) ]);
                
                
                correct_count = 0;
                fail_terminate_count = 0;
            end
        end
    end
    
end