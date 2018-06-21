% Path to wave files
FilePath = '/home/paulo/github/bayeseg/Data';
Files = ['2015.01.30_02.02.56.wav'; '2015.02.02_07.50.49.wav'; '2015.02.08_11.26.39.wav'];

% Divergent color map
LB=flipud(lbmap(256,'BrownBlue'));


%% Figures 7 and 8
    % Reads file
    arquivo = Files(1,:);
    [wave, fs] = audioread(strcat(FilePath, '/', arquivo));        

    % Spectrogram
    N = size(wave, 1);
    [spec fres] = spgram(wave, fs, 0.3, 3, 0.75);
    Freq = linspace(0, size(spec,1)*fres, size(spec, 1));
    iptsetpref('ImshowAxesVisible','on');
    imshow(imadjust(mat2gray((spec))), 'YData', Freq); set(gca, 'YDir', 'Normal');
    daspect auto;
    ax = gca;
    ax.XTick = linspace(0, size(spec,2), 20); ax.XTickLabel = round(linspace(0, N/fs/60, 20),1);
    ax.XTickLabelRotation = 45;
    xlabel('Time (minutes)'); y = ylabel('Frequency (Hz)');
    colormap(LB);   
    colorbar;

%% Figure 9
    t = csvread(strcat(FilePath, '/', '20150202.csv'));

    % Reads file
    arquivo = Files(2,:);
    [wave, fs] = audioread(strcat(FilePath, '/', arquivo));        

    % Spectrogram
    N = size(wave, 1);
    [spec fres] = spgram(wave, fs, 0.3, 3, 0.75);
    secc = (size(wave,1)/fs)/size(spec,2);
    Freq = linspace(0, size(spec,1)*fres, size(spec, 1));
    iptsetpref('ImshowAxesVisible','on');
    imshow(imadjust(mat2gray((spec))), 'YData', Freq); set(gca, 'YDir', 'Normal');
    daspect auto;
    ax = gca;
    ax.XTick = linspace(0, size(spec,2), 20); ax.XTickLabel = round(linspace(0, N/fs/60, 20),1);
    ax.XTickLabelRotation = 45;
    xlabel('Time (minutes)'); y = ylabel('Frequency (Hz)');
    colormap(LB);   
    colorbar;

    for l=1:size(t,2)
        x = t(l)/fs/secc;
        line([x x], [0 6000], 'Color', 'white', 'LineWidth', 1, 'LineStyle', '--');
    end
    set(y, 'position', get(y,'position')-[0.5,0,0]);
    
%% Figure 10
    t = csvread(strcat(FilePath, '/', '20150208.csv'));

    % Reads file
    arquivo = Files(3,:);
    [wave, fs] = audioread(strcat(FilePath, '/', arquivo));        

    % Spectrogram
    N = size(wave, 1);
    [spec fres] = spgram(wave, fs, 0.3, 3, 0.75);
    secc = (size(wave,1)/fs)/size(spec,2);
    Freq = linspace(0, size(spec,1)*fres, size(spec, 1));
    iptsetpref('ImshowAxesVisible','on');
    imshow(imadjust(mat2gray((spec))), 'YData', Freq); set(gca, 'YDir', 'Normal');
    daspect auto;
    ax = gca;
    ax.XTick = linspace(0, size(spec,2), 20); ax.XTickLabel = round(linspace(0, N/fs/60, 20),1);
    ax.XTickLabelRotation = 45;
    xlabel('Time (minutes)'); y = ylabel('Frequency (Hz)');
    colormap(LB);   
    colorbar;

    for l=1:size(t,2)
        x = t(l)/fs/secc;
        line([x x], [0 6000], 'Color', 'white', 'LineWidth', 1, 'LineStyle', '--');
    end
    set(y, 'position', get(y,'position')-[0.5,0,0]);    