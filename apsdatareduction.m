%% Initialize Workspace
clear
clc

%% Read and Sort Files

% get the folder contents
Raw_Data_Path = 'Y:\APS\2021-3_1IDC\WAXS\He_2-2';
Output_Data_Path = 'Y:\APS\2021-3_1IDC\WAXS_extract\pf\He_2-2-As-Is\pole_figure'; % output file directory
d = dir(Raw_Data_Path);
% remove all files (isdir property is 0)
dfolders = d([d(:).isdir]);
% remove '.' and '..'
dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));
dfolders = struct2table(dfolders);

for j = 1:size(dfolders)
    for k = 0:10:350
        Operation_Data_Path = strcat(Raw_Data_Path, "\", char(dfolders{j,1}),"\ge5_mat\"); % current directory
        append_name = strcat("\", char(dfolders{j,1}),"_");
        result = process(Operation_Data_Path, k, append_name, Output_Data_Path);
    end
end


%% Process Data

function output = process(directory, tth, appendname, output_directory)

    cd(directory);
    files1 = dir('**');
    files1(1:2) = [];
    F = struct2table(files1); % convert the struct array to a table
    sortedF = sortrows(F, 'name'); % sort the table by 'name'
    toDelete = not(endsWith(sortedF.name, '.mat')); % Remove unnecessary files
    sortedF(toDelete,:) = [];
    totalFiles = numel(sortedF.name);

    % define tth, 1 = 0 deg, 10 = 90 deg, 19 = 180 deg, 28 = 270 deg
    tth_grid = tth/10 + 1;
    
    % Make list of file address
    parfor i = 1:totalFiles
        Fileaddress(i) = strcat(directory, '\', sortedF.name(i));
    end

    intensity(totalFiles,3000) = zeros;

    % Extract intensity from tables
    parfor i = 1:totalFiles
        try
            Y = load(Fileaddress{i});
            polimg = Y.polimg;
            intensity(i,:) = polimg.intensity_in_tth_grid(tth_grid,:);
        catch
            fprintf(strcat("Error reading files in iteration ", sprintf('%.0f',i), " : ", sortedF.name(i), "\n"));
        end

    end

    save_file_name = strcat(output_directory, appendname, sprintf('%03d %.0f',tth), "deg.csv");
    writematrix(intensity,save_file_name);
    output = "";
end