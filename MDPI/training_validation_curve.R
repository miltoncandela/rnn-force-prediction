create_df <- function(){
     
     path_d <- 'D:/El HDD/R/BRAIN-R/Biomec/Paper/data/'
     n_neurons <- c(4, 8, 16)
     m <- matrix(vector(), 0, 3, dimnames=list(c(), c('N_Neuron', 'T_MAE', 'V_MAE')))
     df <- data.frame(m)
     
     for (n_neuron in n_neurons){
          df_temp <- read.csv(paste0(path_d, 'LSTM_',n_neuron, '_history.csv'))
          df_temp <- df_temp[,grep('mae', colnames(df_temp))]
          df_temp['n'] <- n_neuron
          
          colnames(df_temp) <- c('T_MAE', 'V_MAE', 'N_Neuron')
          df <- rbind(df, df_temp)
     }
     
     df
}

df_to_line <- function(df, name='barplot_biomec'){
     
     pdf('TV_Curve.pdf')
     #windowsFonts(A = windowsFont("Palatino Linotype"))
     
     par(mfrow = c(3, 1), oma = c(4, 1, 1, 1), mar = c(0, 0, 0, 0))
     n_samples <- 500
     
     for (n_neuron in unique(df$N_Neuron)){
             df_neuro <- df[df$N_Neuron==n_neuron, c('T_MAE', 'V_MAE')]
             
             par(mai=c(0,  0.5, 0, 0)) # bottom, left, top, right
             plot(-1, -1, ylim =c(0, 0.3),ylab = paste0('MAE (', n_neuron, ' Neurons)'), xlim = c(0, 25), axes = FALSE)
             lines(1:25, df_neuro[, 'T_MAE'], lwd = 2)
             lines(1:25, df_neuro[, 'V_MAE'], lty = 2, lwd = 2)
             axis(2, at = c(0, 0.10, 0.20, 0.30), labels = c(0, 0.10, 0.20, 0.30))
     }
     axis(1, pos = 0)
     mtext('Epoch', side = 1, cex = 0.75)
     
     par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
     plot(0, 0, type = 'l', bty = 'n', yaxt = 'n', xaxt = 'n')
     
     legend('bottom',legend = c("Training", "Validation"), lwd = 2, xpd = TRUE, horiz = TRUE, lty = c(1, 2),  bty = 'n')
     
     dev.off()
}

df_to_line(create_df())