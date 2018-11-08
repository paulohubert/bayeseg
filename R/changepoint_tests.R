install.packages('changepoint')
install.packages('tuneR')
install.packages('tidyr')
library(tuneR)
library(changepoint)

folder = '/home/paulo/github/bayeseg/Data/'

filelist = list.files(folder)

f = filelist[4]

wav = readWave(paste0(folder, f))
y = wav@left
y = y / max(y)

minlen = 11025
Q = length(y) / minlen

methods = c('AMOC', 'PELT', 'SegNeigh', 'BinSeg')

for(i in 1:length(filelist)) {
  f = filelist[i]
  if(grepl('.wav', f)) {
    wav = readWave(paste0(folder, f))
    y = wav@left
    y = y / max(y)
    init = proc.time()
    tseg = cpt.var(data = y, know.mean = TRUE, mu = 0, method = methods[2], pen.value = 0.1, minseglen = minlen)
    end = proc.time()
    dur = end - init
    teps = as.numeric(dur)[3]
    print(paste0(f,";",teps,";",ncpts(tseg)))
  }
}


summary(tseg)
dur = end - init
dur$elapsed
