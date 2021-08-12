%% Configure the score lists.
% You can copy the data from the excel and paste them at Matlab workspace.
% Then rename the table to 'T' and rename the colName to 'pred' & 'mos'. 
pred = T.pred; % Set the predicted score.
mos = T.mos; % Set the MOS.

%% Calculate PLCC, SRCC and KRCC.
f = polyfit(pred, mos, 3);
fitted_mos = polyval(f, pred);
fitted_PLCC = abs(corr(fitted_mos, mos, 'type', 'Pearson'));

PLCC = abs(corr(pred, mos, 'type', 'Pearson'));
SRCC = abs(corr(pred, mos, 'type', 'Spearman'));
KRCC = abs(corr(pred, mos, 'type', 'Kendall'));

summary = [PLCC,fitted_PLCC,SRCC,KRCC];