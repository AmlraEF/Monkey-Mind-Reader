%% Authors : Spike Girls

% Amira El Fekih, Iani Gayo, Gauri Gupta, Joanna-Svilena Haralampieva 

%% 
function ModelParam = positionEstimatorTraining(training_data)

%Model parameters to learn:
    %A. Average spike rate : avg_spike_300
    %B. Displacement matrix : displacement_matrix
    %C. Baseline firing rate, standard deviation : baseline_across_trials (98x1) and std: std_baseline_angles
    %D. Tuning curve: P(Spike | Angle ) 
    %E: P(Angle | Spike) 
    %F: Counter 
    
%% INITIALISING VARIABLES

TRAIN_SIZE = 100;  
ANGLE_SIZE = 8;
INTERVAL_SIZE = 40; 

%Initialising firing rate matrices 
rate_run_total = zeros(ANGLE_SIZE,98); 
baseline_firing_rates = zeros(98, TRAIN_SIZE, ANGLE_SIZE);
std_neurons = zeros(98,ANGLE_SIZE); 

%Initialising x,y coords 
displacement_vec = zeros(TRAIN_SIZE, ANGLE_SIZE, INTERVAL_SIZE);

   %% A) Average spike rate for first 300ms : avg_spike_300
   
    %Learn average spike rate for first 300ms 
    avg_spike_300 = obtain_avg_spike_rate(training_data); 

    %% B) Displacement Matrix : displacement_matrix
            %Obtained Average firing rates for 300ms onwards 
            %displacement_vec contains magnitude of displacements for every 20ms interval 

    % Obtain Average firing rate for all neurons for 300ms : end , across all trials, angles
    for t_no = 1:TRAIN_SIZE %Iterate through every trial and angle 

        for angle = 1:8 %Iterate through every angle 

            single_trial = training_data(t_no,angle).spikes(:,:); %Obtain spike train for every neuron
            hand_pos = training_data(t_no,angle).handPos(1:2, :); %Hand positions : x y coordinates 
            
             %Saving each x y position for each trial
             x = hand_pos(1, 300:20:end); 
             y = hand_pos(2, 300:20:end); 
                
             %Finding magnitude of displacements for each 20ms interval 
            for index = 1:length(x)-1 %index of 20ms interval

                p2 = [x(index+1), y(index+1)];
                p1 = [x(index), y(index)];

                d = p2 - p1;

                displacement_val = norm(d, 2);

                displacement_vec(t_no, angle, index) = displacement_val; 

            end 
            
            
            
            %% OBTAINING BASELINE FIRING RATE : FOR STANDARDISATION 
            %Obtain single mean baseline rate 
            baseline_fire = mean(single_trial(:,300:end), 2); 
            %Place into big matrix of all baselines according to angle, trial
            baseline_firing_rates(:, t_no, angle) = baseline_fire ;

            %% AVERAGE FIRING RATE FOR ALL NEURONS, FOR SINGLE TRIAL FOR SINGLE ANGLE 
            rates(angle, :) = mean(single_trial(:, 300:end), 2);  %Obtains the average firing rate for each neuron across length of signal

        end 
        
        rates_all(t_no, :,:) = rates; %Saves firing rate each neuron at different angles, for this trial

    end 

    displacement_matrix = squeeze(mean(displacement_vec, 1));  %8x40 %Split up into 40 20ms intervals 
                                                               %Displacement
                                                               %for every
               
                                                               %angle 
    %Moving average filter 
    shady_moving_avg(:, 1) = displacement_matrix(:, 1); 
    for i = 2:40
        shady_moving_avg(:, i) = mean(displacement_matrix(:, 1:i),2) ; %2nd    
    end 
    
    %Actual moving avg
    for angle = 1:8
        real_moving_avg(angle, :) = movmean(displacement_matrix(angle, :), [10 0]); 
    end 
    
                                         
    %% C) Baseline Firing rate : baseline_across_trials (98x1) and std: std_baseline_angles
            %Baseline firing rate for every neuron
    
    % BASELINE FIRING RATE ACROSS ALL ANGLES AND TRIALS:
        % baseline_across_trials
    baseline_across_angles = mean(baseline_firing_rates(:, :, :), 3);  %98x80 For every trial Mean across angles
    baseline_across_trials = mean(baseline_across_angles(:, :), 2); %98x1 : BASELINE FIRING RATE FOR ALL NEURONS

    % OBTAIN STANDARD DEVIATION FOR BASELINE FIRING RATES:
        %std_baseline_angles
    std_baseline = squeeze(std(baseline_firing_rates(:,:,:) , 0 , 2)); %Baseline 98x8
    std_baseline_angles = mean(std_baseline, 2); %Mean to get 98x1 across angles  

    %% D: Tuning curve: P(S|A) : TUNING_CURVE 
    
    %Step 1: Unnormalised tuning curve: 
        %Average across every trial to obtain mean and standard deviation 
        %across all trials for 8 different angles 
        
    %mean_rates is average firing rate for neuron, for every angle : 8x98
    mean_rates = squeeze(mean(rates_all, 1)); % 8x98 : Unnnormalized tuning curve 
    %std_rates = squeeze(std(rates_all, 0, 1)); % Standard deviation 
    
    %Step 2: Normalisation: Subtract min, then divide by max-min
    
    %Obtain min and max variables 
    min_vals = min(mean_rates); 
    max_vals = max(mean_rates); 
    norm_rates = (mean_rates - min_vals) ./ (max_vals - min_vals) ;  
    
    %Step 3: Turning Tuning Curve into probability distribution : P(Spike|Angle)
    sum_vals = sum(norm_rates, 2); %1x8: Sum across all neurons, for every angle
    prob_rates = norm_rates ./ sum_vals ; % P(spike | angle), to get probabilities s.t. the sum is one 
    
    %FINAL TUNING CURVE, ie P(SPIKE | ANGLE) 
    TUNING_CURVE = prob_rates'; %98x8 
    
    %% E) P( A | S) : TRAINING_MATRIX 
    %USING BAYES RULE : P(Angle|Spike) = P(Spike|Angle)P(A) / sum(P(Spike|Angle)P(A))
    psa_times_pa =((1/8) * TUNING_CURVE); %MULTIPLY P(S|A) BY P(A) where P(A) = 1/8, ie tuning curve matrix x 1/8
    norm_prob_vec = sum(psa_times_pa, 2); %SUM(P(S|A)P(A)) 

    TRAINING_MATRIX =  psa_times_pa ./ norm_prob_vec ; %98X8 : P(Angle|Spike)
   
 
    %% PUT TRAINING MATRIX INTO STRUCT
    
    
    ModelParam = struct('Avg_spike_rate', avg_spike_300, 'Displacement_matrix', displacement_matrix, 'Baseline_firing_rate', baseline_across_trials, 'Baseline_std', std_baseline_angles, 'Tuning_curve', TUNING_CURVE, 'TRAINING_MATRIX', TRAINING_MATRIX, 'Counter', 1, 'Moving_avg', real_moving_avg)
    
    
end 


function avg_spike_rates = obtain_avg_spike_rate(training_data)

avg_spike_rates = zeros(98, 8);

    for angle = 1:8

        %Populate all_trials with average spike rate for every trial
        all_trials = zeros(98,100); %all_trials is 98x100

        for trial_nr = 1:TRIAL_SIZE

            %Extract neuronal spikes per trial for time period 1:320
            single_trial = training_data(trial_nr, angle).spikes(:,1:320); %Single_trial is 98x320
%             disp('size of single trial: expect 98x320')
%             disp(size(single_trial))

            %Obtain spike rate for every neuron
            single_spike_rate = mean(single_trial, 2); %single_spike_rate is 98x1
%             disp('size of single spike rate: expect 98x1')
%             disp(size(single_spike_rate))

            %Fill all_trials with single trial 
            all_trials(:, trial_nr) = single_spike_rate; %all_trials is 98x100
%             disp('size of all trials: expect 98x100')
%             disp(size(all_trials))

        end 

       %Mean across all the trials
 
       avg_spike_single_angle = mean(all_trials, 2); %avg spike rate single angle is 98x1
%        disp('size of avg spike single angle: expect 98x1')
%        disp(size(avg_spike_single_angle))
            
       avg_spike_rates(:, angle) = avg_spike_single_angle; %Populate model parameters with average spike rate for one angle 
%        disp('size of avg spike rate: expect 98x8')
%        disp(size(avg_spike_rates))
       
    end 
    
end 


