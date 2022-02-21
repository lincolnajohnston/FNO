%% Read the P3 solver - varyBoronAndWater results from text file with format: results{i}.txt
waterNDID = fopen("../p3solver/waterND.txt");
water_NDs_1d = fscanf(waterNDID, '%f');
fclose(waterNDID);
boronNDID = fopen("../p3solver/boronND.txt");
boron_NDs_1d = fscanf(boronNDID, '%f');
fclose(boronNDID);
fluxg1 = [];
fluxg2 = [];
fluxg1_output = [];
fluxg2_output = [];
water_NDs = [];
boron_NDs = [];
a = [];
numFluxes = 3;
for i=0:1000
    kfileID = fopen(strcat(strcat(strcat("../p3solver/results", num2str(i)),"_k-eff"), ".txt"), 'r');
    if kfileID == -1
            break
        end
    k_vals = fscanf(kfileID, '%f');
    fclose(kfileID);
    for k=1:numFluxes
        disp(strcat("read results ", num2str(i)))
        fluxfileID = fopen(strcat(strcat(strcat(strcat("../p3solver/results", num2str(i)),"-"),num2str(k - 1)), ".txt"), 'r');
        if fluxfileID == -1
            break
        end
        A = fscanf(fluxfileID, '%f');
        if k < numFluxes
            fluxg1 = cat(2, fluxg1, A(2:3:end));
            fluxg2 = cat(2, fluxg2, A(3:3:end));
            fclose(fluxfileID);
            water_val = water_NDs_1d(i + 1);
            boron_val = boron_NDs_1d(i + 1);
            for j=1:400
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
            for j=401:500
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=501:600
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
            for j=601:700
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=701:800
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
            for j=801:900
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=901:1100
                a(i * (numFluxes-1) + k, j,1) = water_val;
            end
            for j=1101:1200
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=1201:1300
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
            for j=1301:1400
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=1401:1500
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
            for j=1501:1600
                a(i * (numFluxes-1) + k,j,2) = boron_val;
            end
            for j=1601:2001
                a(i * (numFluxes-1) + k,j,1) = water_val;
            end
        end
        if k > 1
            fluxg1_output = cat(2, fluxg1_output, A(2:3:end) * k_vals(k));
            fluxg2_output = cat(2, fluxg2_output, A(3:3:end) * k_vals(k));
            %fclose(fluxfileID);
        end
    end
end
fluxg1 = transpose(fluxg1);
fluxg2 = transpose(fluxg2);
group_flux = cat(3, fluxg1, fluxg2);
fluxg1_output = transpose(fluxg1_output);
fluxg2_output = transpose(fluxg2_output);
group_flux_output = cat(3, fluxg1_output, fluxg2_output);
a = cat(3, a, group_flux);
