function imds = load_data(data_path)
    imds = imageDatastore(data_path, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    imds.ReadSize = 64; % batch size
    imds = shuffle(imds);
end