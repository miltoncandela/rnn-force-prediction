# Biomec ours Processing

raw <- read.delim('results/mae.txt', header=FALSE)
feat_order <- c('Z_Leg', 'Y_Leg', 'X_Leg', 'Z_Arm', 'Y_Arm', 'X_Arm')

extract_data <- function(a, i){
     # From array of strings to row of dataframe
     
     # Desired df:
     #    Train    Valid    Test
     # F1
     # F2
     # F3
     
     df <- data.frame(matrix(nrow=6,ncol=3))
     
     row.names(df) <- feat_order
     
     for (n_feat in 1:length(a)){
          df[feat_order[n_feat],] <- as.numeric(strsplit(gsub('\\]| ', '', strsplit(a[n_feat], split = '\\[')[[1]][2]), ',')[[1]])
     }
     df <- cbind(df, i)
     
     colnames(df) <- c('Train', 'Valid', 'Test', 'ID')
     
     df <- cbind('Feat' = rownames(df), df)
     rownames(df) <- 1:nrow(df)
     
     df <- reshape(df, direction = "wide", idvar="ID", timevar="Feat")
     df <- df[, -1]
     df
}

df_pross <- data.frame(matrix(nrow=0, ncol=0))

for (n_comb in 1:(nrow(raw)/7)){
     # Array of n_feats (6) strings
     a_temp <- raw[((n_comb-1)*7+2):((n_comb)*7),]
     
     # Extract info and put it on a df
     df_temp <- extract_data(a_temp, n_comb)
     df_pross <- rbind(df_pross, df_temp)
}


#bp = barplot(apply(df_pross, 1, sum), names.arg=1:24, legend.text = TRUE,
#             ylim = c(0, 10), las = 2, ylab = 'Sum of R^2 for each feature',
#             xlab = 'Iteration number',
#             main = 'Sum of R^2 on each iteration number for each feature')
#text(bp, apply(df_pross, 1, sum), labels = apply(df_pross, 1, sum), pos=3, cex = 0.8)

great_ids <- c(1, 2, 3, 12, 17)
better_ids <- c(21, 19, 23, 22, 7)

df_filt <- df_pross

df_mean <- round(apply(df_filt, 2, mean), 2)
bp = barplot(as.matrix(df_mean), beside=TRUE, names.arg = names(df_mean), 
             las = 2, ylab = 'R^2', angle = 45, ylim = c(-0.4, 1),
             main = 'Average R^2 for each dataset and feature (5 CV)')
text(bp, df_mean, labels = df_mean, pos=3, cex = 0.8)
