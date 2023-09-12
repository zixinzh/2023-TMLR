function [ iout, corr_flag, ...
    regret, cum_regret, regret_exp, cum_regret_exp, ...
    w, index_opt, ...
    W_sel, reward_opt ]...
    = BoBW_lilUCB_to_submit( T, L, w_opt, w_gap_para, reward_distr, alg_para_set  ) 

 
% BOB_lilUCB
%
% Input:
%     K --- no. of opt arms
%     L --- no. of all arms
%     w_opt --- w of opt arms
%     w_gap --- gap of w of opt and subopt arms
%     w_sub --- w_opt - w_gap; % w of subopt arms
%     T --- no. of trials
% Output:
%     mu_exp --- the experimental mean of arms at each time t
%     T_exp --- no. of observations of arms at each time t
%     regret --- true regret at each time t
%     regret_sum --- cumulative regret
%     regret_exp --- expected regret at each time t
%     regret_exp_sum --- expected cumulative regret
%     arm_sel --- selected arms at each time t
%     W_sel --- the true outcome of selected arms at each time t
%     W_opt --- the true outcome of optimal arms at each time t
%     index_opt --- the random locations of optimal items

%% 1. initialize
omega = alg_para_set(1);
sigma = alg_para_set(2);
varepsilon = alg_para_set(3);
beta = alg_para_set(4); 
 
% suboptimal items with the same mean
index_opt = randi([1 L],1);
w = ( w_opt - min(1,w_gap_para) )*ones(1,L);
w(index_opt) = w_opt;
  


regret = -1*ones(1,T);
regret_exp = -1*ones(1,T); 

%% 2. main phase
% 2.1. generate random variables
T_sample = min(T, 2*10^4); 
if strcmp( reward_distr, 'bern' )
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

 

% 2.2. calculate for optimal items  
reward_opt = W_pull( 1:T_sample,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)  
reward_exp_opt = w_opt;   

% 2.3. run for t = 1, ..., L
mu_exp = diag( W_pull(1:L,:) ); % experimental mu 
N_exp = ones(1,L); % number of pulls
Conf_rad = ones(1,L)*5*sigma*(1+sqrt(varepsilon))*sqrt( 2*(1+varepsilon) *log( log( beta+(1+varepsilon) )./omega ) );
U_exp = mu_exp' + Conf_rad; 
regret(1:L) = reward_opt(1:L) - mu_exp; % regret
regret_exp(1:L) = reward_exp_opt - w( 1:L ); % expected regret 

% 2.4. run for t = L+1, ..., T
for t = L+1:T
    % 1: select arms to pull  
    [ ~, U_ord ] = sort(U_exp, 'descend');
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
        clear w_repeat
        % 2.2. calculate for optimal items 
        reward_opt = W_pull( 1:T_sample,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    W_sel = W_pull( t_tmp,arm_sel_tmp  ); % result of selected arms 
    
    % 3: calculate regret
    regret(t) = reward_opt(t_tmp ) - W_sel; % regret
    regret_exp(t) = reward_exp_opt - w( arm_sel_tmp  ); % expected regret 
    
    % 4: update parameters for the pulled item
    mu_exp(arm_sel_tmp) = ( mu_exp(arm_sel_tmp)*N_exp(arm_sel_tmp) + W_sel ) / ( N_exp(arm_sel_tmp) + 1 );
    N_exp(arm_sel_tmp) = N_exp(arm_sel_tmp) + 1;
    
    Conf_rad(arm_sel_tmp) = 5*sigma*(1+sqrt(varepsilon))*sqrt( 2*(1+varepsilon)./N_exp(arm_sel_tmp)*log( log( beta+(1+varepsilon).*N_exp(arm_sel_tmp) )./omega ) );
    U_exp(arm_sel_tmp) = mu_exp(arm_sel_tmp) + Conf_rad(arm_sel_tmp);

 
     
end

% 2.4. output iout
[ ~, mu_ord ] = sort(mu_exp, 'descend');
iout = mu_ord(1);
corr_flag = ( iout == index_opt );

cum_regret = cumsum(regret); % cumulative regret
cum_regret_exp = cumsum(regret_exp); % cumulative regret

end



