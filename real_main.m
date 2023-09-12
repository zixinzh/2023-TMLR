clear
clc
close all

warning('off')


%% 0.load data
dataset_name = 'movielens' % 'pkis2'


if strcmp( dataset_name, 'movielens' )
    load('real_stat_movielens.mat', 'rating_50k');
    data_code_range = {'rating_50k'};
end
if strcmp( dataset_name, 'pkis2' )
    load('real_stat_pkis2.mat', 'data_MAPKAPK5');
    data_code_range = { 'data_MAPKAPK5' };
end


distr_para_set = [ 0.5, 5];
 
%% 1.initialize
T1 = 1;
T0 = 2;
T = T1*10^T0;
no_seed = 2; 
seed_range = 1:no_seed;
disp(['no_seed:', num2str(no_seed) ]);
%%%
if strcmp( dataset_name, 'movielens' )
    reward_distr = 'gaussStd'; % 'gaussStd', 'bern', 'ClipGaussStd','logGaussStd'
end
if strcmp( dataset_name, 'pkis2' )
    reward_distr = 'logGaussStd'; % 'gaussStd', 'bern', 'ClipGaussStd','logGaussStd'
end


disp(['reward_distr:', reward_distr ]);




timetable = zeros(1, no_seed);

algo_short_name = {'BoBW_lilUCB', 'UCB_alpha'  };
algo_folder_name = {'1 BoBW_lilUCB', '2 UCB_alpha'  };
algo_print_name = {'1_BoBW_lilUCB', '2_UCB_alpha' };
%%%%%%%%%%%%%%%%%%%%%%%
sigma = 1;
beta = exp(1);
varepsilon = 0.01;
BOB_omega_range = 9.*(10.^[ -7  -4  -1 ]);%9.*(10.^[ -7 -4 -1 ]);%[ 0.1 0.3 0.5 0.7 0.9];%10.^[ -7 -6 -5 -4 -3 -2 -1 ];
%
num_BOB_omega = length( BOB_omega_range );
BOB_alg_para_set_range = [ BOB_omega_range; [ sigma varepsilon beta ]'*ones( 1, num_BOB_omega ) ];
%%%%%%%%%%%%%%%%%%%%%%%
delta = 10^(-2);
alpha_range = [ 3 4.5 6]; 
%
UCB_alpha_alg_para_set_range = [ alpha_range; [ sigma delta]'*ones( 1, length(alpha_range) ) ];
%%%%%%%%%%%%%%%%%%%%%%% 


cum_regret_exp_all = zeros(no_seed,1);
correct_ratio_all = -1*ones(no_seed,1);
t_stop_all = -1*ones(no_seed,1);
flag_terminate_all = -1*ones(no_seed,1);


time_name = ['T=', num2str(T1), 'e', num2str(T0)];

correct_count = 0;
fail_terminate_count = 0;



disp('algo_name T std_T_stop L data_code alg_para_set - - - correct_ratio fail_terminate_count mean_cum_regret_exp std_cum_regret_exp run_time');



for data_code_raw = data_code_range
    data_code = data_code_raw{1};
    if strcmp( data_code, 'rating_50k' )
        w_vec = rating_50k; %
    end
    if strcmp( data_code, 'data_MAPKAPK5' )
        w_vec = data_MAPKAPK5; %
    end 
    L = length(w_vec); % no. of all arms
    
    for algo_no = [  2 ]
        
        folder_name = [ dataset_name,' fixConf ', time_name,...
            ' ', algo_short_name{algo_no}, '_', num2str(no_seed)  ];
        if exist(folder_name, 'dir')==0
            mkdir(folder_name);
            mkdir([folder_name, '_', num2str(no_seed) ]);
        end
        
        if algo_no == 1
            alg_para_set_range = BOB_alg_para_set_range;
        end
        if algo_no == 2
            alg_para_set_range = UCB_alpha_alg_para_set_range;
        end
        if algo_no == 3
            alg_para_set_range = UCB_E_alg_para_set_range;
        end
        
        for alg_para_set_ind = 1: size( alg_para_set_range, 2 )
            alg_para_set = alg_para_set_range(:,alg_para_set_ind);
            
            if algo_no == 1 
                    simulation_name = [ algo_print_name{algo_no}, ' T=', num2str(T), ' L=', num2str(L),  ...
                        ' data_code=', data_code, ' reward_distr=', reward_distr,...
                        ' alg_para_set=', num2str(alg_para_set'), ...
                        '_algo ', num2str(algo_no)]; 
            end
            if algo_no == 2 
                    simulation_name = [ algo_print_name{algo_no}, ' L=', num2str(L),  ...
                        ' data_code=', num2str(data_code), ' reward_distr=', reward_distr,...
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
                        = real_BoBW_lilUCB( T, L, w_vec, reward_distr, distr_para_set, alg_para_set  ) ;
                end
                if algo_no == 2
                    T_max = max(5*10^5, floor(3*T));
                    [ iout, corr_flag, ...
                        regret, cum_regret, regret_exp, cum_regret_exp, ...
                        w, index_opt, ...
                        W_sel, reward_opt,...
                        t, flag_terminate ]...
                        = real_UCB_alpha( T_max, L, w_vec, reward_distr, distr_para_set, alg_para_set  ) ;
                end 
                
                timetable_tmp = toc(tstart);
                timetable(seed_index) = timetable_tmp;
                
                %% 3.seed save
                if (algo_no == 1) || (algo_no == 3) 
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
                num2str(L), ' ', num2str(data_code), ' ',...
                num2str(alg_para_set'), ' ' ...
                num2str( correct_count/no_seed ), ' ', num2str(fail_terminate_count),' ', ...
                num2str( mean_regret(end) ), ' ', num2str(std_regret(end)), ' ',...
                num2str(ave_time) ]);
            
            
            correct_count = 0;
            fail_terminate_count = 0;
        end
    end
end