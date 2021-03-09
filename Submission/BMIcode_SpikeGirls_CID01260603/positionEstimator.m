%% Authors : Spike Girls

% Amira El Fekih, Iani Gayo, Gauri Gupta, Joanna-Svilena Haralampieva 

%% 

function [x_val, y_val, newParameters] = positionEstimator(test_trial, ModelParam)

single_trial_spikes = test_trial.spikes; 

%% INITIALISING PARAMETERS: TRANSITIONAL PROBABILITIES 

% Transitional probabilities
% gauri_trans_probs = [0.7 0.1 0.02 0.02 0.02 0.02 0.02 0.1; ...
%                0.1 0.7 0.1 0.02 0.02 0.02 0.02 0.02 ;...
%                0.02 0.1 0.7 0.1 0.02 0.02 0.02 0.02 ; ...
%                0.02 0.02 0.1 0.7 0.1 0.02 0.02 0.02; ...
%                0.02 0.02 0.02 0.1 0.7 0.1 0.02 0.02; ...
%                0.02 0.02 0.02 0.02 0.1 0.7 0.1 0.02; ...
%                0.02 0.02 0.02 0.02 0.02 0.1 0.7 0.1; ...
%                0.1 0.02 0.02 0.02 0.02 0.02 0.1 0.7];

%% 1. Identify which direction it most likely corresponds to : come back to counter after 

    %Reset trial, initialise transitional probabilities 
    if length(single_trial_spikes) == 320
         ModelParam.counter = 1;
         
         %Initialise transitional probabilities
         
        angles  = (pi/180) * [30 , 70 , 110, 150, 190, 230, 310, 350];
    
        angles_changed = (pi/180) * [30, 70, 110, 150, -170, -130, -50, -10]; %Between -180-180
    
        std_angles = 0.1 ; %std(angles_changed); 
    
        angles_dif = zeros(8,8) ;
        
        for i = 1:8

            %Take max of two values 
            temp = abs(angles_changed - angles_changed(i));
            x = 5; 
            for j = 1:length(temp)
                    if temp(j) > 180
                       temp(j) = abs(temp(j) - 360); 
                    end 
            end 

            angles_dif(i, :) = temp;  

            %Obtain transitional probabilities
            trans_prob(i, :) = normpdf(0, angles_dif(i,:), std_angles) ;
            ModelParam.trans_prob = trans_prob; 

        end
    end 
    
    %Only perform NN on first 300ms of data 
    if ModelParam.Counter == 1
        [angle_class, distances] = nn_find_angle(single_trial_spikes, ModelParam.Avg_spike_rate); 
        ModelParam.angle_class = angle_class; 
    end 

    ModelParam.Counter = ModelParam.Counter + 1; %Update counter everytime : restart 
    %Save angle class to parameter list 
    
    %% 2. Split up spike rate to 20ms intervals

     firing_rates_vec = [];
     %Obtain firing rate for 20ms intervals   
     for time = 320:20:length(single_trial_spikes)

                    %Firing rates for every 20ms interval 
                    spikes_in_single_interval = single_trial_spikes(:, time-20:time); 
                    
                    %Average firing rate in this interval
                    firing_single_interval = mean(spikes_in_single_interval, 2);
                    
                    firing_rates_vec = [firing_rates_vec, firing_single_interval] ; %Concatenate firing rates 
     
     end 
    
     %Set NaN values to 0 
     firing_rates_vec(isnan(firing_rates_vec)) = 0; 
    
     [no_neurons, length_vec] = size(firing_rates_vec);
     
    %% 3. Select the Magnitude corresponding to the class 

    %NO_INTERVALS = 10; %No of time intervals to consider for average 
    
    mag_vec_all = ModelParam.Displacement_matrix(ModelParam.angle_class, :);
   
    %Method 1: Fixed average magnitude: Using the mean of the first NO_INTERVALS only 
    NO_INTERVALS = 10;
    Magnitude_vec = mag_vec_all(1:NO_INTERVALS); %Obtain magnitude vectors corresponding to angle class
    average_magnitude = mean(Magnitude_vec); %Obtain single value for average displacement 
      
    %Method 2: Moving average 
    %average_magnitude = mean(mag_vec_all(1:length_vec)); 
    
   
     %% 4. Standardise firing rates vector : 
     
     stand_firing_rates_vec = (firing_rates_vec - ModelParam.Baseline_firing_rate) ./ ModelParam.Baseline_std ; %98x8
     
     
     %% 5. Amira's method: A Variation of Population Vector decoding
        %Obtain unit directions and update training matrix using recursive
        %Bayesian filter 
     
     TRAINING_MATRIX = ModelParam.TRAINING_MATRIX; 
     %Do not update training matrix in ModelParam as this is learnt from Training 
     
     for time_step = 1:length_vec
         
  
        %moving_avg_val = ModelParam.Moving_avg(ModelParam.angle_class, time_step); %Obtain moving average value at same index
     
        moving_avg_val = ModelParam.Displacement_matrix(ModelParam.angle_class, time_step) ; 
        %Weight each training matrix by firing rate  
        weighted_training_matrix = TRAINING_MATRIX' * stand_firing_rates_vec(:, time_step); %8x98 * 98x1 --> 8x1

        %sum_training_matrix = sum

        %Step 3: Normalise by 98 neurons 
        norm_weighted_training_matrix = weighted_training_matrix; %/ 98; 
        %norm_weighted_training_matrix = weighted_training_matrix / sum(weighted_training_matrix); 

        [max_vals, idx] = max(norm_weighted_training_matrix) ;
  
        
        %Step 4: Multiply each normalised weighted probability by angle unit vectors
        angles  = (pi/180) * [30 , 70 , 110, 150, 190, 230, 310, 350]; 

        %UNIT VECTOR DIRECTIONS : x = 1cos(theta), y = 1sin(theta)

        angle_unit_vectors = [cos(angles)', sin(angles)']; % 8X2 col_1 = x, col_2 = y
    
        final_directions = (norm_weighted_training_matrix)' * angle_unit_vectors;%1*8 x 8x2 --> 1x2 (x,y at this time step)
        
        unit_directions_amira(:,time_step) = average_magnitude*(final_directions / norm(final_directions)); %2x1
        
        %Update training matrix 
        TRAINING_MATRIX = bayesian_filter(ModelParam.Tuning_curve, ModelParam.trans_prob, TRAINING_MATRIX); 
      
     end 
     
     %Obtaining trajectory 
     [traj_x, traj_y] = find_trajectory(unit_directions_amira); 
     
     %Only obtain final value as "x, y" coordinates
     x_val = traj_x(end);
     y_val = traj_y(end); 
     
     %% 6. Update ModelParameters
     newParameters = ModelParam;
     
end



function [angle_class, distances] = nn_find_angle(single_test_trial, model_param_avg_spike)

test = single_test_trial;

%Obtain spike rate for test 
spike_rate = mean(test, 2); %spike rate is 98x1

%Obtain difference matrix
diff_matrix = model_param_avg_spike - repmat(spike_rate, [1,8]);
%disp(size(diff_matrix))

distances = zeros(1,8); 

for i = 1:8
    
    distances(i) = norm(diff_matrix(:, i), 2); 
    
end 

disp(distances)
 
%Obtain index of minimum distance

[min_val, idx] = min(distances); 

%disp(idx)

angle_class = idx; 

end 

function TRAINING_MATRIX_NEW = bayesian_filter(TUNING_CURVE, trans_prob,TRAINING_MATRIX)

%% BAYESIAN STEPS

%transpose the trans_prob so that each column = (X_t-1) 
%SUM ACROSS EACH POSSIBLE ANGLE AT TIME-1

%SUM_across all A_t-1, P(X | X_1) * P(A_t-1 | S_t-1) 
summation_term = trans_prob' * TRAINING_MATRIX'; %8x8 8X98 --> 8X98 

%P(S|A) * P(A) 
TRAINING_MATRIX_NEW = TUNING_CURVE .* summation_term' ; %98X8 * 9X98 --> 98x8 

ALPHA = sum(TRAINING_MATRIX_NEW, 2); %sum across the columns : 98x1 

%P(Angle | Spike1... Spike_t-1) 
TRAINING_MATRIX_NEW = TRAINING_MATRIX_NEW ./ ALPHA;


end

function [traj_x, traj_y] = find_trajectory(unit_directions_amira)

    
    [row, length_interval] = size(unit_directions_amira) ;
    
    traj_x = [];
    traj_y = [];

    %Start 
    traj_x = [traj_x 0]; %1 x1
    traj_y = [traj_y 0]; %1 x1

    
    for i = 1:length_interval

        traj_x = [traj_x traj_x(i)+unit_directions_amira(1, i)]; 
        traj_y = [traj_y traj_y(i)+unit_directions_amira(2, i)]; 

    end 

end 
