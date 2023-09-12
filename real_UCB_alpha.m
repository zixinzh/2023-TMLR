function [ iout, corr_flag, ...
    regret, cum_regret, regret_exp, cum_regret_exp, ...
    w, index_opt, ...
    W_sel, reward_opt,...
    t, flag_terminate ]...
    = UCB_alpha_movielens( T, L, w_vec, reward_distr, distr_para_set, alg_para_set  ) 

 
% UCB_alpha 
% Reference: 
%     Bridging the gap between regret minimization and best arm identification, with application to A/B tests
%
% Input: 
%     L --- no. of all arms
%     w_vec --- w of arms 
%     T --- no. of trials
% Output:
%     mu_exp --- the experimental mean of arms at each time t
%     T_exp --- no. of observations of arms at each time t
%     regret --- true regret at each time t
%     regret_sum --- cumulative regret
%     regret_exp --- expected regret at each time t
%     regret_exp_sum --- expected cumulative regret 
%     W_sel --- the true outcome of selected arms at each time t
%     W_opt --- the true outcome of optimal arms at each time t
%     index_opt --- the random locations of optimal items

%% 1. initialize
alpha = alg_para_set(1); 
sigma = alg_para_set(2);
delta = alg_para_set(3);

 

 

% shuffle items
w = w_vec(randperm(length(w_vec)));
[w_opt,index_opt] = max(w); 
 


regret = -1*ones(1,T);
regret_exp = -1*ones(1,T);  

%% 2. main phase
% 2.1. generate random variables
T_sample = min(T, 3*10^3);
if strcmp( reward_distr, 'bern' )
    T_sample = max(T, 3*10^4);
    w_repeat = repmat(w,T_sample,1); % all w repeat
    w_repeat = reshape(w_repeat,1,T_sample*L);
    W_pull = binornd(1,w_repeat);
    W_pull = reshape(W_pull,T_sample,L);
    clear w_repeat
end 

if strcmp( reward_distr, 'gaussStd' ) 
    w_repeat = repmat(w,T_sample,1); % all w repeat 
    W_pull = normrnd(w_repeat,1);
    clear w_repeat
end

if strcmp( reward_distr, 'logGaussStd' )
    w_raw = w;
    w = log(w_raw);
    w_repeat = repmat(w,T_sample,1); % all w repeat 
    W_pull = normrnd(w_repeat,1);
    clear w_repeat
end

reward_min = distr_para_set(1);
reward_max = distr_para_set(2);
    
if strcmp( reward_distr, 'ClipGaussStd' )
    w_repeat = repmat(w,T_sample,1); % all w repeat 
    W_pull = normrnd(w_repeat,1);
    W_pull = max( reward_min, W_pull);
    W_pull = min( reward_max, W_pull );
    clear w_repeat
end 

% 2.2. calculate for optimal items  
if strcmp( reward_distr, 'logGaussStd' )
    reward_opt = exp( W_pull( 1:T_sample,index_opt ) ); % result of true opt arms (CAN BE OUT OF THE LOOP)  
    reward_exp_opt = exp( w_opt );
else
    reward_opt = W_pull( 1:T_sample,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)  
    reward_exp_opt = w_opt;   
end
 
% 2.3. run for t = 1, ..., N_0*L 
%%% to have the sqrt holds, we need 3(log N_exp )^2 > delta   
%%% <=>   log N_exp > sqrt( delta/3 )   
%%% <=>   N_exp > exp( sqrt( delta/3 ) )
N_0 = ceil( exp( sqrt(delta/3) ) ); 
N_exp = N_0*ones(1,L); % number of pulls
mu_exp = -1*ones(1,L); % experimental mu 
for i = 1:L
    tmp_index_range = N_0*(i-1)+1:N_0*i; 
    tmp_W_range = W_pull(tmp_index_range,i);
    mu_exp(i) =  mean( tmp_W_range ); % experimental mu 
%     regret( tmp_index_range ) = reward_opt( tmp_index_range ) - tmp_W_range; % regret
%     regret_exp( tmp_index_range ) = reward_exp_opt - w( i ); % expected regret   
    if strcmp( reward_distr, 'logGaussStd' )
        regret(tmp_index_range) = reward_opt(tmp_index_range) - exp(tmp_W_range); % regret
        regret_exp(tmp_index_range) = reward_exp_opt - exp(w( i)); % expected regret 
    else
        regret(tmp_index_range) = reward_opt(tmp_index_range) -  tmp_W_range; % regret
        regret_exp(tmp_index_range) = reward_exp_opt - w( i ); % expected regret 
    end 
    
end 
Conf_rad = sigma*sqrt( 2./N_0 * log( 3*(log(N_0 )).^2./delta )  );
Conf_rad_alpha = alpha.*Conf_rad;
U_exp_alpha = mu_exp + Conf_rad_alpha; 
U_exp = mu_exp + Conf_rad; 
L_exp = mu_exp - Conf_rad;  
t = N_0*L;
flag_terminate = 0;
  


% 2.4. run for t = N_0*L+1, ... 
while t < T 
    t = t + 1;
    % 1: select arms to pull  
    [ ~, U_ord ] = sort(U_exp_alpha, 'descend');
    arm_sel_tmp  = U_ord(1);    
    
    % 2: pull selected and optimal arms
    t_tmp = mod(t-1,T_sample)+1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if t_tmp == 1
        clear W_pull
        if strcmp( reward_distr, 'bern' )
            w_repeat = repmat(w,T_sample,1); % all w repeat
            w_repeat = reshape(w_repeat,1,T_sample*L);
            W_pull = binornd(1,w_repeat);
            W_pull = reshape(W_pull,T_sample,L); 
        end 
        if strcmp( reward_distr, 'gaussStd' )
            w_repeat = repmat(w,T_sample,1); % all w repeat
            W_pull = normrnd(w_repeat,1); 
        end  
        if strcmp( reward_distr, 'logGaussStd' )
            w_repeat = repmat(w,T_sample,1); % all w repeat 
            W_pull = normrnd(w_repeat,1);
        end
        if strcmp( reward_distr, 'ClipGaussStd' )
            w_repeat = repmat(w,T_sample,1); % all w repeat 
            W_pull = normrnd(w_repeat,1);
            W_pull = max( reward_min, W_pull);
            W_pull = min( reward_max, W_pull ); 
        end
        clear w_repeat
        % 2.2. calculate for optimal items 
        if strcmp( reward_distr, 'logGaussStd' )
            reward_opt = exp( W_pull( 1:T_sample,index_opt ) ); % result of true opt arms (CAN BE OUT OF THE LOOP)   
        else
            reward_opt = W_pull( 1:T_sample,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)  
        end 
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    W_sel = W_pull( t_tmp,arm_sel_tmp  ); % result of selected arms 
    
    % 3: calculate regret
    if strcmp( reward_distr, 'logGaussStd' )
        regret(t) = reward_opt(t_tmp ) - exp(W_sel); % regret
        regret_exp(t) = reward_exp_opt - exp(w( arm_sel_tmp )); % expected regret 
    else
        regret(t) = reward_opt(t_tmp ) - W_sel; % regret
        regret_exp(t) = reward_exp_opt - w( arm_sel_tmp ); % expected regret 
    end  
    
    % 4: update parameters for the pulled item
    mu_exp(arm_sel_tmp) = ( mu_exp(arm_sel_tmp)*N_exp(arm_sel_tmp) + W_sel ) / ( N_exp(arm_sel_tmp) + 1 );
    N_exp(arm_sel_tmp) = N_exp(arm_sel_tmp) + 1;
    Conf_rad(arm_sel_tmp) = sigma*sqrt( 2./N_exp(arm_sel_tmp)* log( 3*(log(N_exp(arm_sel_tmp))).^2./delta )  );
    Conf_rad_alpha(arm_sel_tmp) = alpha*Conf_rad(arm_sel_tmp);
    U_exp_alpha(arm_sel_tmp) = mu_exp(arm_sel_tmp) + Conf_rad_alpha(arm_sel_tmp);
    U_exp(arm_sel_tmp) = mu_exp(arm_sel_tmp) + Conf_rad(arm_sel_tmp);
    L_exp(arm_sel_tmp) = mu_exp(arm_sel_tmp) - Conf_rad(arm_sel_tmp); 
    
    % 5. check stopping rule
    [ ~, index_max_mu ] = max( mu_exp );
    max_Lcb = L_exp(index_max_mu);
    Ucb_tmp = U_exp; 
    Ucb_tmp(index_max_mu) = max_Lcb-1;
    max_Ucb_except = max( Ucb_tmp );
    
 
    
    if max_Lcb > max_Ucb_except
        flag_terminate = 1;
        break
    end 
end

% 2.5. output iout
[ ~, mu_ord ] = sort(mu_exp, 'descend');
iout = mu_ord(1);
corr_flag = ( iout == index_opt );



cum_regret = cumsum(regret); % cumulative regret
cum_regret_exp = cumsum(regret_exp); % cumulative regret

end



