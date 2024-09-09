library(readxl)
library(dplyr)
library(tidyverse)

#load the dataframe with all parameters, "Path" needs to be replaced with your path
all_subs_excel <- read_excel("Path/all_subs_excel.xlsx")


#list of electrodes: 
elecs = all_subs_excel$channel[1:19]
elecs2 = rep(elecs, each = 3)
electrodes = rep(elecs2, 12)

#prepare dataframe for statistics per electrode and model:
beta_val <- rep(NaN, 19*3*12)
std_val <- rep(NaN, 19*3*12)
t_val <- rep(NaN, 19*3*12)
CIL <- rep(NaN, 19*3*12)
CIU <- rep(NaN, 19*3*12)
p_val <- rep(NaN, 19*3*12)
cond <- rep(c("Intercept", "Clin_vs_Ctrl", "AD_vs_FTD"), 19*12)
dep_var <- rep(c('rel_delta', 'rel_theta', 'rel_alpha', 'rel_beta', 'rel_gamma', 'offset', 'exponent',
               'adj_delta', 'adj_theta', 'adj_alpha', 'adj_beta', 'adj_gamma'), each = 19*3)

my_aper_stats = data.frame(dep_var, electrodes, cond, beta_val, std_val, t_val, p_val, CIL, CIU)


#desired contrast: clinical vs control (i.e., C vs A & F), A vs F
#use Reverse Helmert: Compare level one to the last two and level 2 to level 3
all_subs_excel$Group = factor(all_subs_excel$Group, levels = c("C", "A", "F")); 
rev_h = matrix(c(2,-1,-1,0,1,-1), nrow=3, ncol=2);
contrasts(all_subs_excel$Group)=rev_h
contrasts(all_subs_excel$Group)


#loop through the electrodes, calculate rel_delta model, get statistics:
row_nr = 0
rows = seq(from=1, to=57, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(rel_delta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate relative theta model
rows = seq(from=58, to=114, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(rel_theta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate relative alpha model
rows = seq(from=115, to=171, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(rel_alpha~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate relative beta model
rows = seq(from=172, to=228, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(rel_beta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate relative gamma model
rows = seq(from=229, to=285, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(rel_gamma~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate offset model
rows = seq(from=286, to=342, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(offset~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate exponent model
rows = seq(from=343, to=399, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(exponent~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate adjusted delta model
rows = seq(from=400, to=456, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(adj_delta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}


#loop through the electrodes, calculate adjusted theta model
rows = seq(from=457, to=513, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(adj_theta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}


#loop through the electrodes, calculate adjusted alpha model
rows = seq(from=514, to=570, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(adj_alpha~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}


#loop through the electrodes, calculate adjusted beta model
rows = seq(from=571, to=627, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(adj_beta~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}

#loop through the electrodes, calculate adjusted gamma model
rows = seq(from=628, to=684, by=3)

for (r in rows) { 
  elec = my_aper_stats$electrodes[r]
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- all_subs_excel %>% filter(grepl(elec, channel))
  
  model1 = lm(adj_gamma~Group, data = elec_df) 
  model_CIs = confint(model1, c(1,2,3), level = 0.95)
  
  #in dataframe, first row intercept: 
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[1,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[1,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[1,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[1,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[1,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[1,4]
  
  #second row Clin_vs_Ctrl:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[2,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[2,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[2,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[2,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[2,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[2,4]
  
  #third row AD_vs_FTD:
  row_nr = row_nr + 1
  my_aper_stats[row_nr, "beta_val"] <- summary(model1)$coefficients[3,1]
  my_aper_stats[row_nr, "std_val"] <- summary(model1)$coefficients[3,2]
  my_aper_stats[row_nr, "t_val"] <- summary(model1)$coefficients[3,3]
  my_aper_stats[row_nr, 'CIL'] <- model_CIs[3,1]
  my_aper_stats[row_nr, 'CIU'] <- model_CIs[3,2]
  my_aper_stats[row_nr, "p_val"] <- summary(model1)$coefficients[3,4]
  
}


#unmute the next line to save the linear model results, 'Path' needs to be 
#replaced with your path

#write.table(my_aper_stats, file = "Path/lin_models_results.csv", sep = ";", dec = ",", row.names = FALSE, col.names = TRUE) 

#note:before the next steps in the analysis in python, I loaded the "lin_models_results.csv"
#file into an excel table (in the python script: lin_models_results.xlsx)





#####################################################################################

#%%Exploratory Analysis: 
#In advance of this script, I created a separate file based on the "all_subs_excel"
#file, that only contains the data from the AD and FTD groups and includes the MMSE
#score of each individual as a column

#replace path with your path
my_df <- read_excel("path/AD_FTD_all_data.xlsx")

#prepare dataframe for statistics per electrode:
beta <- rep(NaN, 19*12*4)
STD <- rep(NaN, 19*12*4)
t_value <- rep(NaN, 19*12*4)
p_value <- rep(NaN, 19*12*4)
CIL <- rep(NaN, 19*12*4)
CIU <- rep(NaN, 19*12*4)
significance <- rep(NaN, 19*12*4)
Band <- rep(c("rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
              "adj_delta", "adj_theta", "adj_alpha", "adj_beta", "adj_gamma", 
              'offset', 'exponent'), each=19*4)
cond <- rep(c("Intercept", "Group_AD_vs_FTD",
              "MMSE", "AD_vs_FTDxMMSE"), 19*12)

#list of electrodes: 
elecs = my_df$channel[1:19]
electrodes1 = rep(elecs, each=4)
electrodes = rep(electrodes1, 12)

power_stats = data.frame(electrodes,
                         Band,
                         cond,
                         beta,
                         STD,
                         t_value,
                         p_value,
                         significance)
power_stats <- power_stats[order(power_stats$electrodes),]
row.names(power_stats) <- NULL


my_df <- my_df[order(my_df$channel),]

row_nr = 0
rows = seq(from=1, to=1121, by=59)

for (r in elecs) { 
  elec = r
  row_nr = row_nr  + 1
  
  #get dataframe containing values of all subjects for the electrode
  elec_df <- my_df %>% filter(grepl(elec, channel))
  elec_df$Group = factor(elec_df$Group);
  
  model_delta <- lm(rel_delta ~ Group*MMSE, data = elec_df)
  model_theta <- lm(rel_theta ~ Group*MMSE, data = elec_df)
  model_alpha <- lm(rel_alpha ~ Group*MMSE, data = elec_df)
  model_beta <- lm(rel_beta ~ Group*MMSE, data = elec_df)
  model_gamma <- lm(rel_gamma ~ Group*MMSE, data = elec_df)
  model_delta_adj <- lm(adj_delta ~ Group*MMSE, data = elec_df)
  model_theta_adj <- lm(adj_theta ~ Group*MMSE, data = elec_df)
  model_alpha_adj <- lm(adj_alpha ~ Group*MMSE, data = elec_df)
  model_beta_adj <- lm(adj_beta ~ Group*MMSE, data = elec_df)
  model_gamma_adj <- lm(adj_gamma ~ Group*MMSE, data = elec_df)
  model_off <- lm(offset ~ Group*MMSE, data = elec_df)
  model_exp <- lm(exponent ~ Group*MMSE, data = elec_df)
  
  
  #in dataframe, first row delta intercept: 
  power_stats[row_nr, "beta"] <- summary(model_delta)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_delta)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta)$coefficients[1,4]
  int1 = confint(model_delta, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  
  #second row delta AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_delta)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta)$coefficients[2,4]
  int1 = confint(model_delta, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #third row delta MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_delta)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta)$coefficients[3,4]
  int1 = confint(model_delta, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #fourth row delta AD_vs_FTDxMMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_delta)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta)$coefficients[4,4]
  int1 = confint(model_delta, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #fifth row theta intercept: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_theta)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta)$coefficients[1,4]
  int1 = confint(model_theta, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #sixth row theta Ad_v_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_theta)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta)$coefficients[2,4]
  int1 = confint(model_theta, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #7th row theta MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_theta)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta)$coefficients[3,4]
  int1 = confint(model_theta, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #8th row theta AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_theta)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta)$coefficients[4,4]
  int1 = confint(model_theta, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #9th row alpha intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha)$coefficients[1,4]
  int1 = confint(model_alpha, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #10th row alpha AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha)$coefficients[2, 2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha)$coefficients[2,4]
  int1 = confint(model_alpha, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #alpha MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha)$coefficients[3,4]
  int1 = confint(model_alpha, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #alpha AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha)$coefficients[4,4]
  int1 = confint(model_alpha, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #beta intercept: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_beta)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta)$coefficients[1,4]
  int1 = confint(model_beta, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #beta AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_beta)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta)$coefficients[2,4]
  int1 = confint(model_beta, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #beta MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_beta)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta)$coefficients[3,4]
  int1 = confint(model_beta, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #beta AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_beta)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta)$coefficients[4,4]
  int1 = confint(model_beta, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #gamma intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma)$coefficients[1,4]
  int1 = confint(model_gamma, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #gamma AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma)$coefficients[2,4]
  int1 = confint(model_gamma, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #gamma MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma)$coefficients[3,4]
  int1 = confint(model_gamma, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #gamma AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma)$coefficients[4,4]
  int1 = confint(model_gamma, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted delta intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta_adj)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_delta_adj)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta_adj)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta_adj)$coefficients[1,4]
  int1 = confint(model_delta_adj, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted delta AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta_adj)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_delta_adj)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta_adj)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta_adj)$coefficients[2,4]
  int1 = confint(model_delta_adj, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted delta MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta_adj)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_delta_adj)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta_adj)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta_adj)$coefficients[3,4]
  int1 = confint(model_delta_adj, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted delta AD_vs_FTDxMMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_delta_adj)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_delta_adj)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_delta_adj)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_delta_adj)$coefficients[4,4]
  int1 = confint(model_delta_adj, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted theta intercept: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta_adj)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_theta_adj)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta_adj)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta_adj)$coefficients[1,4]
  int1 = confint(model_theta_adj, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted theta Ad_v_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta_adj)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_theta_adj)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta_adj)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta_adj)$coefficients[2,4]
  int1 = confint(model_theta_adj, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted theta MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta_adj)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_theta_adj)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta_adj)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta_adj)$coefficients[3,4]
  int1 = confint(model_theta_adj, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted theta AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_theta_adj)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_theta_adj)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_theta_adj)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_theta_adj)$coefficients[4,4]
  int1 = confint(model_theta_adj, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted alpha intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha_adj)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha_adj)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha_adj)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha_adj)$coefficients[1,4]
  int1 = confint(model_alpha_adj, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted alpha AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha_adj)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha_adj)$coefficients[2, 2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha_adj)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha_adj)$coefficients[2,4]
  int1 = confint(model_alpha_adj, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted alpha MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha_adj)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha_adj)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha_adj)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha_adj)$coefficients[3,4]
  int1 = confint(model_alpha_adj, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted alpha AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_alpha_adj)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_alpha_adj)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_alpha_adj)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_alpha_adj)$coefficients[4,4]
  int1 = confint(model_alpha_adj, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted beta intercept: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta_adj)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_beta_adj)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta_adj)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta_adj)$coefficients[1,4]
  int1 = confint(model_beta_adj, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted beta AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta_adj)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_beta_adj)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta_adj)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta_adj)$coefficients[2,4]
  int1 = confint(model_beta_adj, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted beta MMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta_adj)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_beta_adj)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta_adj)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta_adj)$coefficients[3,4]
  int1 = confint(model_beta_adj, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted beta AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_beta_adj)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_beta_adj)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_beta_adj)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_beta_adj)$coefficients[4,4]
  int1 = confint(model_beta_adj, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted gamma intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma_adj)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma_adj)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma_adj)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma_adj)$coefficients[1,4]
  int1 = confint(model_gamma_adj, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted gamma AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma_adj)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma_adj)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma_adj)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma_adj)$coefficients[2,4]
  int1 = confint(model_gamma_adj, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted gamma MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma_adj)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma_adj)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma_adj)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma_adj)$coefficients[3,4]
  int1 = confint(model_gamma_adj, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #adjusted gamma AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_gamma_adj)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_gamma_adj)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_gamma_adj)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_gamma_adj)$coefficients[4,4]
  int1 = confint(model_gamma_adj, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #offset intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_off)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_off)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_off)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_off)$coefficients[1,4]
  int1 = confint(model_off, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #offset AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_off)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_off)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_off)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_off)$coefficients[2,4]
  int1 = confint(model_off, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #offset MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_off)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_off)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_off)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_off)$coefficients[3,4]
  int1 = confint(model_off, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #offset AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_off)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_off)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_off)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_off)$coefficients[4,4]
  int1 = confint(model_off, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #exponent intercept:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_exp)$coefficients[1,1]
  power_stats[row_nr, "STD"] <- summary(model_exp)$coefficients[1,2]
  power_stats[row_nr, "t_value"] <- summary(model_exp)$coefficients[1,3]
  power_stats[row_nr, "p_value"] <- summary(model_exp)$coefficients[1,4]
  int1 = confint(model_exp, 1, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #exponent AD_vs_FTD:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_exp)$coefficients[2,1]
  power_stats[row_nr, "STD"] <- summary(model_exp)$coefficients[2,2]
  power_stats[row_nr, "t_value"] <- summary(model_exp)$coefficients[2,3]
  power_stats[row_nr, "p_value"] <- summary(model_exp)$coefficients[2,4]
  int1 = confint(model_exp, 2, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #exponent MMSE: 
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_exp)$coefficients[3,1]
  power_stats[row_nr, "STD"] <- summary(model_exp)$coefficients[3,2]
  power_stats[row_nr, "t_value"] <- summary(model_exp)$coefficients[3,3]
  power_stats[row_nr, "p_value"] <- summary(model_exp)$coefficients[3,4]
  int1 = confint(model_exp, 3, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
  #exponent AD_vs_FTDxMMSE:
  row_nr = row_nr + 1
  power_stats[row_nr, "beta"] <- summary(model_exp)$coefficients[4,1]
  power_stats[row_nr, "STD"] <- summary(model_exp)$coefficients[4,2]
  power_stats[row_nr, "t_value"] <- summary(model_exp)$coefficients[4,3]
  power_stats[row_nr, "p_value"] <- summary(model_exp)$coefficients[4,4]
  int1 = confint(model_exp, 4, level = 0.95)
  power_stats[row_nr, "CIL"] <- int1[1]
  power_stats[row_nr, "CIU"] <- int1[2]
  
}


write.table(power_stats, file = "path/Exploratory_Results.csv", sep = ";", dec = ".", row.names = FALSE, col.names = TRUE) 




