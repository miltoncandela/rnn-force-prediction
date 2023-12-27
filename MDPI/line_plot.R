# Lineplot 1 file

create_df <- function(){
     
     path_d <- 'D:/PycharmProjects/Biomec-DigitalTwin/RNN_Dataset/'
     folder <- 'E-25_U-8_B-128_S-5'
     
     m <- matrix(vector(), 0, 3 * 4 + 1, dimnames=list(c(), c('X', 'R2_X', 'R2_Y', 'R2_Z', 'R2_Avg',
                                                              'Pearson_X', 'Pearson_Y', 'Pearson_Z', 'Pearson_Avg',
                                                              'PVal_X', 'PVal_Y', 'PVal_Z', 'PVal_Avg')))
     df <- data.frame(m)
     
     for (samp in c('Down', 'Up')){
          curr_path <- paste0(path_d, samp, '_Sample/', folder, '/csv/')
          for (file in list.files(curr_path)){
               df_temp <- read.csv(paste0(curr_path, file))
               
               df_temp['samp'] <- strsplit(strsplit(file, '_')[[1]][1], '-')[[1]][2]
               df_temp['loss'] <- strsplit(strsplit(file, '_')[[1]][2], '-')[[1]][2]
               df_temp['laye'] <- strsplit(strsplit(strsplit(file, '_')[[1]][3], '-')[[1]][2], '\\.')[[1]][1]
               
               df <- rbind(df, df_temp)
          }
     }
     
     df <- df[, -grep('PVal', colnames(df))]
     colnames(df)[1] <- 'Dataset'
     
     print(df)
     
     df
}

df_to_line <- function(df, name='barplot_biomec'){
     dimensions <- c('X', 'Y', 'Z', 'Avg')
     df <- df[, c(paste(c('R2', 'Pearson'), 'Avg', sep='_'), 'Dataset', 'loss', 'laye', 'samp')]
     datasets <- c('Training', 'Validation', 'Testing')
     metrics <- c('R2', 'Pearson')
     letter <- c('a)', 'b)')
     #png('Lineplot.png', height=540, width = 1080)
     
     
     #pdf('Lineplot.pdf', family = "Palatino")
     
     #windowsFonts(A = windowsFont("Palatino Linotype"))
     lb = 0.3
     plot(-1, -1, ylim =c(lb, 1), ylab = 'Performance metrics',
          xlab = 'Loss function', xlim = c(0, 18), axes= FALSE)
     axis(2)
     
     mult <- 0
     
     for (samp in unique(df$samp)){
             
          df_sub <- df[df$samp == samp,]
          
          for (metric in metrics){
               
               n_col <- 2
               for (laye in unique(df$laye)){
                    for (i in 1:length(datasets)){
                         ini <- ((i-1)*3 + 1) + mult
                         
                         serie <- ini:(ini + 2)
                         lines(serie, df_sub[df_sub$laye==laye & df_sub$Dataset == datasets[i], paste(metric, 'Avg', sep='_')],
                               col = n_col, lty = 1, pch = which(metrics==metric), type='b')
                         axis(1, at = serie, labels = toupper(unique(df$loss)), las =2)
                         
                         text(serie[2], lb + 0.05, datasets[i])
                    }
                    n_col <- n_col + 1
               }
               
               legend('topright', legend = toupper(unique(df$laye)), lty = 1,
                      col = 2:(2 + 3), bty = 'n')
               legend('topleft', legend = metrics, pch = 1:2, bty = 'n')
               #  , inset = c(-0.3, 0)
               #bty = 'n',
               #legend('bottomright', legend = c('Training', 'Validation', 'Testing'), lty = 1:3, col = 1, pch = 0)
               
          }
          if (samp == 'down'){
               abline(v = serie[3] + 0.5, lty = 2)
          } 
          
          mult <- 9
          
     }
     
     text(c(5, 14), lb + 0.1, c('Down Sampling', 'Up Sampling'))
     
     #dev.off()
     
}

# par(mar = c(3, 3, 3, 8), xpd = TRUE)
df_to_line(create_df())

#df_to_latex(create_df())
#df_to_barplot(create_df())
#df <- create_df()
#plot(df[df$laye=='gru',]$R2_Avg, type = 'l', xaxt="n", axes = FALSE)
#axis(2)

#axis(1, at = 1:6, labels = paste(1:6, 'a', sep = '_'),las=2)

#axis(1, at = c(-2,2),labels = paste(2, 'a'))

#df <- df[df$Dataset=='Testing',]
#df <- df[,-1]
#df <- df[order(df$samp,df$laye),]

