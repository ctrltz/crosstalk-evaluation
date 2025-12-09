function rift_save_epochs(ft_path, preproc_path, save_path, tag_type, random_phases)

addpath(ft_path);

tag_folder = sprintf('%s/tag_type_%d_random_phases_%d', save_path, tag_type, random_phases);
fprintf("Tag folder: %s\n", tag_folder);
mkdir_if_not_exists(tag_folder);

output_folder = sprintf('%s/preproc', tag_folder);
fprintf("Saving preprocessed data to: %s\n", output_folder);
mkdir_if_not_exists(output_folder);

ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14];

% Loop over each ID and call the function
for i = 1:length(ids)
    subj_id = ids(i);
    fprintf('subj_id = %d, tag_type = %d, random_phases = %d\n', subj_id, tag_type, random_phases);

    % load data
    data = fetch_clean_data(preproc_path, subj_id); % function from original code
    cfg = [];
    cfg.latency = [0.2 1.2];

    % onestim
    fprintf("onestim\n");
    trials = make_trial_selection(data, 'numstim', 1, 'tag_type', tag_type, 'random_phases', random_phases, 'freq1', 60); % function from original code
    cfg.trials = trials;
    data_onestim = ft_selectdata(cfg, data);
    save(sprintf('%s/sub%03d-data_onestim.mat', output_folder, subj_id), 'data_onestim')

    % twostim only has random phases
    if (random_phases)
        fprintf("twostim\n");
        trials = make_trial_selection(data, 'numstim', 2, 'tag_type', tag_type, 'random_phases', 1, 'use_phasetag', 1, 'freq1', 60, 'freq2', 60);
        cfg.trials = trials;
        data_twostim = ft_selectdata(cfg, data);
        save(sprintf('%s/sub%03d-data_twostim.mat', output_folder, subj_id), 'data_twostim')
    end
end

end
