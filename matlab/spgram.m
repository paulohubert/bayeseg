function [spectro, fres] = spgram(signal, fs, tres, nww, overlap)
% Power spectrum estimation based on Welch's method
% 
%
% fs - sample rate in Hz
% tres - time resolution for the spectrogram in seconds
% nww - number of Welch windows to use in each time bin
% overlap - proportion of overlap in Welch windows, \in [0,1)

% @TODO: treat use of "floor" to avoid it
% @TODO: implement better way of receiving arguments, via struct

    N = size(signal, 1);
    
    if N == 1
        signal = signal';
        N = size(signal, 1);
    end

    if(nargin ==1) 
        fs = 11025;
        tres = floor(N / fs) / 1425; % Default = 1425 time bins
        nww = 4; 
    elseif(nargin == 2)
        tres = floor(N / fs) / 1425; % Default = 1425 time bins
        nww = 4;
    elseif(nargin == 3)
        nww = 4;
    end
    
    % Number of time bins
    n_time_bins = N / (tres * fs);
    
    % Number of data points in each bin
    npoints_time_bin = floor(N / n_time_bins);
    
    % Number of points in each Welch window
    npoints_welch = floor(npoints_time_bin / (nww-(nww-1)*overlap));
    
    % Stepsize to calculate start of each Welch window
    stepsize_welch = floor(npoints_welch*(1-overlap));
   
    % Next power of 2 from the length of welch window
    NNFT = 2^nextpow2(npoints_welch);
    
    % We can calculate the maximum frequency we can capture
    max_freq = fs / 2;

    % With this info we are able to calculate the frequency resolution,
    % based on the size in seconds of the signal which will be fourier
    % transformed
    %fres = 1 / (NNFT/2 * fs);    
    fres = max_freq / (NNFT / 2);
    
    % Main loop: separates each window with tres seconds of signal,
    % generate welch windows and average periodogram
    for i=1:n_time_bins
        window = signal((1 + (i-1)*npoints_time_bin):(i*npoints_time_bin));
        avg_pgram = zeros(NNFT/2,1);
        for j = 1:nww
            % Limits for Welch window
            i0 = 1 + (j-1)*stepsize_welch;
            i1 = min(i0 + npoints_welch, npoints_time_bin);
            wwindow = window(i0:i1,1);
            
            % Calculating DFT
            pgram = abs(fft(wwindow, NNFT)/npoints_welch);
            
            avg_pgram = avg_pgram + pgram(1:NNFT/2,1) / nww;
        end
        
        if i == 1
            spectro = avg_pgram;
        else
            spectro = [spectro avg_pgram];
        end
    end
end