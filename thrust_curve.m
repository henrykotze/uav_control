%% Parameters
fit_order = 3;
calc_scalar = true;

%% Print Data
fprintf("\nOrder of fitted function: %d\n\n", fit_order);

%% Motor Data

% Full sweeps
pwm = [1000:25:1725, 1950];
m1_data = [0.58 0.94 1.31 1.81 2.22 2.9 3.4 4.08 4.735 5.44 6.17 6.89 7.715 8.52 9.15 10.1 10.93 11.84 12.7 13.27 14.04 14.7 15.52 16.2 16.8 17.6 18.51 19.29 20.04 20.58 27.4] / 4;

% Plot
figure;
plot(pwm, m1_data);
%legend('Honeybee Thrust');
title('Raw Thrust Values per motor');


%% Fit Model
pwm_full = 1000:1:2000;
scale_val = 2000;

m1_p_scaled = polyfit(pwm / scale_val, m1_data, fit_order);
m1_p = apply_scaling(m1_p_scaled, scale_val);
m1_fit = polyval(m1_p, pwm_full);

% Print
print_coeffs(1, m1_p);

% Plot
figure;
hold on;
plot(pwm_full, m1_fit);
plot(pwm, m1_data, 'o');
hold off;
%legend('Honeybee Thrust per motor Fitted');
title('Honeybee thrust per motor');
xlabel('PWM Value');
ylabel('Thurst (N)');


%% Helper functions
function poly_scaled = apply_scaling(poly, scalar)
    poly_scaled = poly;
    for i = 1:length(poly)
        poly_scaled(i) = poly_scaled(i) / scalar^(length(poly) - i);
    end
end


function print_coeffs(motor_num, coeffs)
    fprintf("Motor %d: ", motor_num);
    for i = 1:length(coeffs)
       fprintf("%+20.16f", coeffs(i));
       if i < length(coeffs)
           fprintf("*x^%d", length(coeffs) - i);
       else
           fprintf("\n");
       end
    end
end